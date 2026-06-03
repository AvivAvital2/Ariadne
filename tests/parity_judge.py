"""LLM-as-judge parity harness (Catalog transition Phase 3.3).

Compares legacy and new doc outputs head-to-head and asks an LLM to rate
each pair on a 5-point scale. Aggregates verdicts into a per-doc-type
summary the user uses to decide whether to flip
``OrchestratorConfig.catalog_only_generator`` to True by default.

Workflow:

1. Capture legacy outputs (one-off):
    OPENAI_API_KEY=... uv run python tests/capture_legacy_fixtures.py \\
        --files <files...> --output-dir tests/fixtures/legacy_outputs

2. Run the new path on the same files (with the flag enabled) and write
   outputs to ``tests/fixtures/new_outputs/``.

3. Run this harness:
    OPENAI_API_KEY=... uv run python tests/parity_judge.py \\
        --legacy-dir tests/fixtures/legacy_outputs \\
        --new-dir tests/fixtures/new_outputs

The judge prompt asks for a 5-point verdict (much_better / better / same /
worse / much_worse) plus a one-line reason. The CLI exits 1 if any single
doc is rated "much_worse" or the average is below "same".
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from collections import Counter
from pathlib import Path

import httpx
from rich.console import Console
from rich.table import Table

# Allow running as `python tests/parity_judge.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

console = Console()


# 5-point scale, ordered worst → best for averaging.
VERDICT_SCORE: dict[str, int] = {
    'much_worse': -2,
    'worse': -1,
    'same': 0,
    'better': 1,
    'much_better': 2,
}
SCORE_VERDICT: dict[int, str] = {v: k for k, v in VERDICT_SCORE.items()}


JUDGE_SYSTEM = '''\
You are an expert technical writer evaluating two versions of a generated
documentation file for the same source code. The two versions come from
different generation pipelines; your job is to judge whether the NEW
version is better, worse, or equivalent in usefulness to a developer.

Focus on:
- Factual coverage of the source (does it explain everything important?)
- Concrete vs generic prose (specific to the code vs filler)
- Correct identification of public APIs / classes / functions
- Absence of hallucinations / wrong claims

Output exactly one line:
VERDICT: <much_better|better|same|worse|much_worse> — <one short reason>
'''


JUDGE_USER_TEMPLATE = '''\
Source description: {description}

=== LEGACY (existing pipeline) ===
{legacy}

=== NEW (catalog-driven pipeline) ===
{new}

Rate the NEW version relative to the LEGACY one. Output ONLY the single
VERDICT line as instructed.
'''


def _extract_verdict(response: str) -> tuple[str, str]:
    """Parse the judge's `VERDICT:` line. Returns (verdict, reason)."""
    m = re.search(r'VERDICT:\s*([a-z_]+)\s*[—-]\s*(.+)', response.strip())
    if m and m.group(1) in VERDICT_SCORE:
        return m.group(1), m.group(2).strip()
    # Fallback: if no parseable verdict, mark as 'same' with a notice.
    return 'same', f'(judge response unparseable: {response[:80]})'


async def _call_judge(
    client: httpx.AsyncClient, *, system: str, user: str, model: str,
) -> str:
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ],
    }
    if model.startswith('gpt-5'):
        payload['max_completion_tokens'] = 200
    else:
        payload['max_tokens'] = 200
        payload['temperature'] = 0.0
    response = await client.post('/chat/completions', json=payload)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']


def _pair_files(legacy_dir: Path, new_dir: Path) -> list[tuple[Path, Path]]:
    """Match files by basename across the two directories."""
    pairs: list[tuple[Path, Path]] = []
    legacy_files = {f.name: f for f in legacy_dir.glob('*.md')}
    new_files = {f.name: f for f in new_dir.glob('*.md')}
    common = sorted(set(legacy_files) & set(new_files))
    for name in common:
        pairs.append((legacy_files[name], new_files[name]))
    only_legacy = set(legacy_files) - set(new_files)
    only_new = set(new_files) - set(legacy_files)
    if only_legacy:
        console.print(
            f'[yellow]not paired (legacy only): {sorted(only_legacy)}[/yellow]'
        )
    if only_new:
        console.print(
            f'[yellow]not paired (new only): {sorted(only_new)}[/yellow]'
        )
    return pairs


async def run(
    legacy_dir: Path,
    new_dir: Path,
    *,
    model: str,
    api_key: str,
) -> tuple[int, dict[str, list[tuple[str, str, str]]]]:
    """Returns (exit_code, results_by_doc_type)."""
    pairs = _pair_files(legacy_dir, new_dir)
    if not pairs:
        console.print('[red]no matching pairs found[/red]')
        return 1, {}

    base_url = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }
    results: dict[str, list[tuple[str, str, str]]] = {}

    async with httpx.AsyncClient(
        base_url=base_url, headers=headers, timeout=120.0,
    ) as client:
        for legacy_path, new_path in pairs:
            description = legacy_path.stem
            legacy = legacy_path.read_text(encoding='utf-8')
            new = new_path.read_text(encoding='utf-8')
            user = JUDGE_USER_TEMPLATE.format(
                description=description, legacy=legacy, new=new,
            )
            try:
                resp = await _call_judge(
                    client, system=JUDGE_SYSTEM, user=user, model=model,
                )
            except httpx.HTTPError as e:
                console.print(f'[red]judge failed for {legacy_path.name}: {e}[/red]')
                continue
            verdict, reason = _extract_verdict(resp)
            doc_type = legacy_path.stem.rsplit('.', 1)[-1] if '.' in legacy_path.stem else '?'
            results.setdefault(doc_type, []).append(
                (legacy_path.name, verdict, reason),
            )
            console.print(f'[cyan]{legacy_path.name}[/cyan] → {verdict} — {reason}')

    return 0, results


def _print_summary(results: dict[str, list[tuple[str, str, str]]]) -> int:
    """Render summary table + return exit-code per quality gate."""
    table = Table(title='Parity Judge — Summary')
    table.add_column('doc_type')
    table.add_column('n', justify='right')
    table.add_column('avg score', justify='right')
    table.add_column('worst')
    table.add_column('distribution')

    overall_scores: list[int] = []
    saw_much_worse = False
    for doc_type, rows in sorted(results.items()):
        if not rows:
            continue
        scores = [VERDICT_SCORE[v] for _name, v, _reason in rows]
        overall_scores.extend(scores)
        worst = min(scores)
        if SCORE_VERDICT[worst] == 'much_worse':
            saw_much_worse = True
        avg = sum(scores) / len(scores)
        dist = Counter(v for _name, v, _reason in rows)
        dist_str = ', '.join(f'{k}={dist[k]}' for k in (
            'much_better', 'better', 'same', 'worse', 'much_worse',
        ) if dist.get(k))
        table.add_row(
            doc_type,
            str(len(rows)),
            f'{avg:+.2f}',
            SCORE_VERDICT[worst],
            dist_str,
        )

    console.print()
    console.print(table)

    avg_overall = sum(overall_scores) / max(len(overall_scores), 1)
    console.print(f'\nOverall average score: [bold]{avg_overall:+.2f}[/bold]')

    if saw_much_worse:
        console.print('[red]GATE FAIL: at least one doc rated much_worse.[/red]')
        return 1
    if avg_overall < 0:
        console.print(
            '[red]GATE FAIL: overall avg below same. Investigate before flipping default.[/red]'
        )
        return 1
    console.print('[green]GATE PASS: average ≥ same and no much_worse.[/green]')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog='parity_judge', description=__doc__)
    parser.add_argument('--legacy-dir', required=True, type=Path,
        help='Directory of legacy doc fixtures (capture_legacy_fixtures output)')
    parser.add_argument('--new-dir', required=True, type=Path,
        help='Directory of new-pipeline doc outputs')
    parser.add_argument('--model', default=os.environ.get('ARIADNE_JUDGE_MODEL', 'gpt-5.2'),
        help='LLM model to use as judge (default: gpt-5.2)')
    args = parser.parse_args()

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        console.print('[red]Set OPENAI_API_KEY to run parity_judge[/red]')
        return 1
    if not args.legacy_dir.exists():
        console.print(f'[red]missing: {args.legacy_dir}[/red]')
        return 1
    if not args.new_dir.exists():
        console.print(f'[red]missing: {args.new_dir}[/red]')
        return 1

    rc_pre, results = asyncio.run(run(
        args.legacy_dir, args.new_dir, model=args.model, api_key=api_key,
    ))
    if rc_pre != 0:
        return rc_pre
    return _print_summary(results)


if __name__ == '__main__':
    sys.exit(main())
