"""Evolving test for `ariadne testimonials` — the best-of reader CLI.

One test grows through the reader's demands: empty store → friendly message;
seeded → highest-first with Q/A/score/permalink; --limit caps; --export copies
the stored images out. Drives a tmp store dir, never a real .ariadne/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import testimonials
from cli.status import cmd_testimonials


def _seed(base: Path, score: int, q: str, a: str, *,
          with_image: bool = False, permalink: bool = True) -> None:
    testimonials.record(
        testimonials.local_dir(base),
        question=q, answer=a, score=score,
        duration_seconds=2.0, asked_at=f'2026-06-14T00:00:{score:02d}Z',
        permalink=f'https://slack.example/p{score}' if permalink else None,
        images=[b'\x89PNGpayload'] if with_image else [],
    )


def test_testimonials_cli_shows_top_then_limits_and_exports(tmp_path: Path, capsys) -> None:
    base = str(tmp_path)

    # Demand 1 — empty store → friendly empty-state, exit 0.
    assert cmd_testimonials(argparse.Namespace(dir=base, limit=20, export=None)) == 0
    assert 'No testimonials' in capsys.readouterr().out

    # Demand 2 — seeded entries print highest-first with Q/A/score/permalink.
    _seed(tmp_path, 9, 'how does caching work?', 'It uses an LRU.', with_image=True)
    _seed(tmp_path, 6, 'what is X?', 'X is Y.')
    _seed(tmp_path, 4, 'no-link question', 'no-link answer', permalink=False)
    assert cmd_testimonials(argparse.Namespace(dir=base, limit=20, export=None)) == 0
    out = capsys.readouterr().out
    assert out.index('how does caching work?') < out.index('what is X?')   # 9 before 6
    assert 'It uses an LRU.' in out
    assert '9/10' in out and 'https://slack.example/p9' in out
    assert 'no-link question' in out          # a testimonial with no permalink still prints

    # Demand 3 — --limit caps the list.
    assert cmd_testimonials(argparse.Namespace(dir=base, limit=1, export=None)) == 0
    out1 = capsys.readouterr().out
    assert 'how does caching work?' in out1 and 'what is X?' not in out1

    # Demand 4 — --export copies the stored images out for the showcase.
    dest = tmp_path / 'showcase'
    assert cmd_testimonials(argparse.Namespace(dir=base, limit=20, export=str(dest))) == 0
    pngs = list(dest.glob('*.png'))
    assert len(pngs) == 1 and pngs[0].read_bytes() == b'\x89PNGpayload'

    # Demand 5 — --export-html writes a self-contained showcase page (the chosen
    # 'cards' skin): a full HTML doc with the questions rendered and any diagram
    # embedded inline — not the terminal listing.
    html_file = tmp_path / 'show.html'
    assert cmd_testimonials(argparse.Namespace(dir=base, limit=20, export=None,
                                               export_html=str(html_file))) == 0
    page = html_file.read_text()
    assert page.startswith('<!DOCTYPE html>') and '</html>' in page
    assert 'how does caching work?' in page           # a seeded question, rendered
    assert 'data:image/png;base64,' in page           # the seeded diagram embedded
    assert 'Wrote' in capsys.readouterr().out         # confirmation, not the Q/A listing
