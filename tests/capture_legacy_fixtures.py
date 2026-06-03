"""Capture legacy-pipeline outputs as frozen fixtures (Catalog Phase 3).

Runs the legacy ``DocGenerator.generate_for_module`` path on each input
Python file and writes every generated doc to disk, organized by file +
doc_type. The resulting fixtures are then used as ground truth for the
LLM-judge harness (``tests/parity_judge.py``).

Usage:

    OPENAI_API_KEY=... uv run python tests/capture_legacy_fixtures.py \\
        --files docgen/orchestrator.py docgen/generator.py \\
        --output-dir tests/fixtures/legacy_outputs

Per Catalog transition Phase 3.4, the user runs this once on a curated
set of ~20 representative files (dataclass / attrs / ABC / async / …).
The output is then committed and used as the parity baseline.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from rich.console import Console

# Allow running as `python tests/capture_legacy_fixtures.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docgen._legacy_analyzer import SourceAnalyzer
from docgen.generator import DocGenerator, GeneratorConfig

console = Console()


def slugify(rel_path: str) -> str:
    """Stable filename slug from a relative path."""
    return rel_path.replace('/', '__').replace('.', '_')


async def capture_one(
    file: Path,
    output_dir: Path,
    *,
    doc_types: tuple[str, ...],
    model: str,
    api_key: str,
) -> int:
    """Run the legacy path on `file`, write each doc to `output_dir`.

    Returns count of docs written.
    """
    analyzer = SourceAnalyzer()
    try:
        metadata = analyzer.analyze_file(file)
    except SyntaxError as e:
        console.print(f'[red]SyntaxError in {file}: {e}[/red]')
        return 0

    gen = DocGenerator(
        config=GeneratorConfig(model=model, api_key=api_key, doc_types=doc_types),
        analyzer=analyzer,
    )
    written = 0
    async with gen:
        # type: ignore[arg-type] — DocGenerator accepts a tuple of DocType literals.
        docs = await gen.generate_for_module(metadata, doc_types=doc_types)  # type: ignore[arg-type]
        for d in docs:
            out_path = output_dir / f'{slugify(str(file))}.{d.doc_type}.md'
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(d.content, encoding='utf-8')
            console.print(f'[green]wrote[/green] {out_path}')
            written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        prog='capture_legacy_fixtures',
        description=__doc__,
    )
    parser.add_argument('--files', nargs='+', required=True,
        help='Python files to run through the legacy generator')
    parser.add_argument('--output-dir', required=True, type=Path,
        help='Directory to write per-doc fixtures')
    parser.add_argument('--model', default=os.environ.get('ARIADNE_MODEL', 'gpt-5.2'),
        help='LLM model to use (default: gpt-5.2)')
    parser.add_argument('--types', default='explanation,architecture',
        help='Comma-separated doc types (default: explanation,architecture)')
    args = parser.parse_args()

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        console.print('[red]Set OPENAI_API_KEY to run capture_legacy_fixtures[/red]')
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    doc_types = tuple(t.strip() for t in args.types.split(','))

    files = [Path(f).resolve() for f in args.files]
    missing = [f for f in files if not f.exists()]
    if missing:
        for m in missing:
            console.print(f'[red]missing[/red] {m}')
        return 1

    async def _run() -> int:
        total = 0
        for f in files:
            console.print(f'[cyan]capturing[/cyan] {f}')
            total += await capture_one(
                f, args.output_dir,
                doc_types=doc_types, model=args.model, api_key=api_key,
            )
        return total

    written = asyncio.run(_run())
    console.print(f'\n[bold green]Done.[/bold green] {written} docs written to {args.output_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
