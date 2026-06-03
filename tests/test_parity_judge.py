"""Unit tests for the parity-judge utility (Catalog Phase 3.3).

The script is user-runnable end-to-end (LLM-driven). These tests cover
the deterministic pieces: verdict parsing, file pairing, and the quality
gate logic.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make tests/ importable so we can pull in the script as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent))


class TestExtractVerdict:
    def test_clean_response_is_parsed(self) -> None:
        from parity_judge import _extract_verdict

        v, reason = _extract_verdict(
            'VERDICT: better — covers more public methods and adds context',
        )
        assert v == 'better'
        assert 'public methods' in reason

    def test_unparseable_response_falls_back_to_same(self) -> None:
        from parity_judge import _extract_verdict

        v, reason = _extract_verdict('looks fine to me')
        # Fallback prevents the harness from crashing on a broken judge response.
        assert v == 'same'
        assert 'unparseable' in reason

    def test_much_worse_recognized(self) -> None:
        from parity_judge import _extract_verdict

        v, _ = _extract_verdict(
            'VERDICT: much_worse — drops the entire dependency analysis',
        )
        assert v == 'much_worse'

    def test_hyphen_separator_works(self) -> None:
        """Real LLM responses sometimes use ASCII hyphen instead of em-dash."""
        from parity_judge import _extract_verdict

        v, reason = _extract_verdict('VERDICT: same - tied on coverage')
        assert v == 'same'
        assert 'tied' in reason

    def test_verdict_buried_in_multiline_response_is_extracted(self) -> None:
        """Some judges preface their verdict with reasoning even when told
        to output only the line. The harness must still find it.
        """
        from parity_judge import _extract_verdict

        v, _ = _extract_verdict(
            'Looking at both versions...\n\n'
            'The legacy is shorter but the new one covers more.\n'
            'VERDICT: better — adds method coverage\n'
        )
        assert v == 'better'

    def test_unknown_verdict_word_falls_back(self) -> None:
        """Imagine the judge writes 'VERDICT: ok — fine' — 'ok' isn't in
        VERDICT_SCORE so we must NOT crash; fallback to 'same' with a
        notice is the contract. (A KeyError in summary aggregation would
        be a real bug.)
        """
        from parity_judge import _extract_verdict

        v, reason = _extract_verdict('VERDICT: ok — fine')
        assert v == 'same'
        assert 'unparseable' in reason or 'ok' in reason

    def test_empty_string_does_not_raise(self) -> None:
        from parity_judge import _extract_verdict

        v, _ = _extract_verdict('')
        assert v == 'same'


class TestPairFiles:
    def test_only_files_present_in_both_dirs_are_paired(
        self, tmp_path: Path,
    ) -> None:
        from parity_judge import _pair_files

        legacy = tmp_path / 'legacy'
        new = tmp_path / 'new'
        legacy.mkdir()
        new.mkdir()

        (legacy / 'a.explanation.md').write_text('legacy a')
        (legacy / 'b.explanation.md').write_text('legacy b')
        (legacy / 'only_legacy.md').write_text('legacy only')
        (new / 'a.explanation.md').write_text('new a')
        (new / 'b.explanation.md').write_text('new b')
        (new / 'only_new.md').write_text('new only')

        pairs = _pair_files(legacy, new)

        names = {p[0].name for p in pairs}
        assert names == {'a.explanation.md', 'b.explanation.md'}

    def test_pair_is_legacy_then_new_in_order(
        self, tmp_path: Path,
    ) -> None:
        """If the implementation accidentally swaps the order, downstream
        rendering of "legacy" vs "new" would be misleading.
        """
        from parity_judge import _pair_files

        legacy = tmp_path / 'L'
        new = tmp_path / 'N'
        legacy.mkdir()
        new.mkdir()
        (legacy / 'x.md').write_text('LEGACY-CONTENT')
        (new / 'x.md').write_text('NEW-CONTENT')

        pairs = _pair_files(legacy, new)
        assert len(pairs) == 1
        assert pairs[0][0].read_text() == 'LEGACY-CONTENT'
        assert pairs[0][1].read_text() == 'NEW-CONTENT'

    def test_subdirectories_are_not_descended(
        self, tmp_path: Path,
    ) -> None:
        """Per design, the harness pairs flat files. If the impl ever
        switched to rglob it would silently start matching nested files
        with potentially conflicting basenames — this pins the contract.
        """
        from parity_judge import _pair_files

        legacy = tmp_path / 'L'
        new = tmp_path / 'N'
        (legacy / 'sub').mkdir(parents=True)
        (new / 'sub').mkdir(parents=True)
        (legacy / 'sub' / 'nested.md').write_text('nested-legacy')
        (new / 'sub' / 'nested.md').write_text('nested-new')

        pairs = _pair_files(legacy, new)
        assert pairs == []


class TestSummaryGate:
    def test_much_worse_fails_gate(self, capsys) -> None:
        from parity_judge import _print_summary

        results = {
            'explanation': [
                ('a.md', 'much_worse', 'drops everything'),
                ('b.md', 'better', 'more concrete'),
            ],
        }
        rc = _print_summary(results)
        assert rc == 1
        out = capsys.readouterr().out
        assert 'GATE FAIL' in out

    def test_avg_below_same_fails_gate(self, capsys) -> None:
        from parity_judge import _print_summary

        results = {
            'explanation': [
                ('a.md', 'worse', 'less detail'),
                ('b.md', 'worse', 'less detail'),
            ],
        }
        rc = _print_summary(results)
        assert rc == 1

    def test_avg_at_or_above_same_passes_gate(self, capsys) -> None:
        from parity_judge import _print_summary

        results = {
            'explanation': [
                ('a.md', 'same', 'tied'),
                ('b.md', 'better', 'stronger'),
            ],
        }
        rc = _print_summary(results)
        assert rc == 0

    def test_empty_results_passes(self, capsys) -> None:
        from parity_judge import _print_summary

        rc = _print_summary({})
        assert rc == 0

    def test_score_weighting_rejects_average_below_zero(self, capsys) -> None:
        """Weighting math: much_worse=-2, worse=-1, same=0, better=+1,
        much_better=+2. An impl bug that swaps signs or treats 'better'
        as -1 would flip a green run red. Pin the math.
        """
        from parity_judge import _print_summary

        # avg = (+2 + -1) / 2 = +0.5 → green
        rc = _print_summary({
            'explanation': [
                ('a.md', 'much_better', 'much better'),
                ('b.md', 'worse', 'slightly worse'),
            ],
        })
        assert rc == 0

        # avg = (+1 + -2) / 2 = -0.5 → red, AND much_worse alone fails
        rc = _print_summary({
            'explanation': [
                ('a.md', 'better', 'better'),
                ('b.md', 'much_worse', 'drops content'),
            ],
        })
        assert rc == 1

    def test_much_worse_fails_even_when_average_is_positive(self, capsys) -> None:
        """A single much_worse must be a fail, even if the rest of the
        sample averages positive — this is the absolute floor of the gate.
        """
        from parity_judge import _print_summary

        rc = _print_summary({
            'explanation': [
                ('a.md', 'much_better', 'great'),
                ('b.md', 'much_better', 'great'),
                ('c.md', 'much_better', 'great'),
                ('d.md', 'much_worse', 'drops critical content'),
            ],
        })
        assert rc == 1
        out = capsys.readouterr().out
        assert 'much_worse' in out


class TestVerdictScoreIntegrity:
    """The contract: every verdict in VERDICT_SCORE must round-trip via
    SCORE_VERDICT. If someone adds a verdict to one without the other,
    summary rendering crashes silently with KeyError.
    """

    def test_verdict_score_inverse_round_trips(self) -> None:
        from parity_judge import SCORE_VERDICT, VERDICT_SCORE

        for v, score in VERDICT_SCORE.items():
            assert SCORE_VERDICT[score] == v
        assert len(VERDICT_SCORE) == len(SCORE_VERDICT)
