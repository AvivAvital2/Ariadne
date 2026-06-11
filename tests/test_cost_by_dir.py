"""Tier 2 of the dry-run explorer — per-directory cost decomposition.

``cost_by_directory`` decomposes the generate-phase estimate
(``pricing.estimate_cost``) per directory and file, so the explorer can
show the real cost of each subtree. It targets the **baseline** (caching
off), which is linear per file, and reuses ``estimate_generate_by_doc_type``
as the pricing oracle — the per-dir numbers ARE the dry-run's numbers.

Grown as one evolving ``test_cost``, then split into the focused tests
below. Fixtures are synthetic: neutral paths, stubbed token hooks, and the
real ``claude-opus-4-8`` rate (5/25 per 1M) so costs are computable by
hand. No filesystem access — with ``input_tokens_for`` stubbed the
estimator never reads the (fictional) files.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from docgen.cost_by_dir import cost_by_directory, format_by_dir_table
from docgen.pricing import CHARS_PER_TOKEN, estimate_cost

# claude-opus-4-8 rates: $5/1M input, $25/1M output.
MODEL = 'claude-opus-4-8'
IN_RATE, OUT_RATE = 5.0, 25.0
BASE = Path('/proj')
PY_TYPES = ('explanation', 'architecture', 'qa')  # all supported for python


def _hooks(content=1000, overhead=100, output=200):
    """Stub the three calibration hooks with constant, known token counts."""
    return dict(
        input_tokens_for=lambda _p: content,
        output_tokens_for=lambda _dt, _lang: output,
        prompt_overhead_for=lambda _dt: overhead,
    )


def test_cost_per_file():
    files = [(BASE / 'src' / 'a.py', 4000)]
    nodes = cost_by_directory(files, BASE, PY_TYPES, MODEL, **_hooks())

    ingestion_per_type = 1000 * IN_RATE / 1_000_000              # 0.005
    gen_per_type = (100 * IN_RATE + 200 * OUT_RATE) / 1_000_000  # 0.0055
    n = len(PY_TYPES)

    leaf = nodes['src/a.py']
    assert leaf.docs == n
    assert {d.doc_type for d in leaf.by_type} == set(PY_TYPES)
    assert all(d.count == 1 for d in leaf.by_type)
    assert all(d.gen_cost == pytest.approx(gen_per_type) for d in leaf.by_type)
    assert leaf.ingestion_cost == pytest.approx(ingestion_per_type * n)
    assert leaf.total == pytest.approx((ingestion_per_type + gen_per_type) * n)
    # total == ingestion + Σ gen_cost
    assert leaf.total == pytest.approx(
        leaf.ingestion_cost + sum(d.gen_cost for d in leaf.by_type)
    )


def test_cost_doc_type_split():
    # yaml's language curation supports only 'explanation' — the other two
    # requested types are filtered out (inherited from estimate_cost).
    nodes = cost_by_directory(
        [(BASE / 'conf' / 'app.yaml', 500)], BASE, PY_TYPES, MODEL, **_hooks(),
    )
    leaf = nodes['conf/app.yaml']
    assert [d.doc_type for d in leaf.by_type] == ['explanation']
    assert leaf.docs == 1


def test_cost_empty_node():
    # A file whose language supports NONE of the requested types still
    # builds a node — zero docs, zero cost, no ingestion (no calls).
    nodes = cost_by_directory(
        [(BASE / 'conf' / 'empty.yaml', 100)], BASE, ('architecture', 'qa'), MODEL,
        **_hooks(),
    )
    leaf = nodes['conf/empty.yaml']
    assert leaf.docs == 0
    assert leaf.by_type == ()
    assert leaf.total == 0.0
    assert leaf.ingestion_cost == 0.0


def test_cost_aggregates_up():
    tree = [
        (BASE / 'src' / 'a.py', 4000),
        (BASE / 'src' / 'api' / 'b.py', 4000),
        (BASE / 'src' / 'api' / 'c.py', 4000),
    ]
    nodes = cost_by_directory(tree, BASE, PY_TYPES, MODEL, **_hooks())

    api = nodes['src/api']
    assert api.total == pytest.approx(
        nodes['src/api/b.py'].total + nodes['src/api/c.py'].total
    )
    assert api.docs == nodes['src/api/b.py'].docs + nodes['src/api/c.py'].docs
    src = nodes['src']
    assert src.total == pytest.approx(nodes['src/a.py'].total + api.total)
    assert src.ingestion_cost == pytest.approx(
        nodes['src/a.py'].ingestion_cost + api.ingestion_cost
    )
    assert src.docs == nodes['src/a.py'].docs + api.docs


def test_cost_parity_with_dry_run_baseline():
    # The load-bearing invariant: the decomposition sums to the dry-run's
    # own baseline estimate over the same files (caching off ⇒ additive).
    tree = [
        (BASE / 'src' / 'a.py', 4000),
        (BASE / 'src' / 'api' / 'b.py', 4000),
        (BASE / 'src' / 'api' / 'c.py', 4000),
    ]
    nodes = cost_by_directory(tree, BASE, PY_TYPES, MODEL, **_hooks())
    baseline = estimate_cost(
        tree, PY_TYPES, MODEL, caching_enabled=False, **_hooks(),
    ).baseline_cost_usd

    leaf_sum = sum(
        nodes[k].total for k in ('src/a.py', 'src/api/b.py', 'src/api/c.py')
    )
    assert nodes['.'].total == pytest.approx(baseline)
    assert leaf_sum == pytest.approx(baseline)


def test_cost_ingestion_dominates():
    # The vendored-installer case: huge content, few docs — re-ingested
    # content swamps scaffolding + output, so ingestion ≫ Σ gen_cost.
    nodes = cost_by_directory(
        [(BASE / 'vendor' / 'bundle.js', 9_000_000)], BASE, PY_TYPES, MODEL,
        **_hooks(content=100_000, overhead=100, output=200),
    )
    leaf = nodes['vendor/bundle.js']
    assert leaf.ingestion_cost > 10 * sum(d.gen_cost for d in leaf.by_type)


def test_cost_no_rate_model():
    # A model absent from LLM_PRICING prices at $0 (mirrors estimate_cost's
    # rates-None path) but still builds nodes with doc counts.
    nodes = cost_by_directory(
        [(BASE / 'src' / 'a.py', 4000)], BASE, PY_TYPES, 'no-such-model', **_hooks(),
    )
    leaf = nodes['src/a.py']
    assert leaf.docs == len(PY_TYPES)
    assert leaf.total == 0.0
    assert leaf.ingestion_cost == 0.0
    assert all(d.gen_cost == 0.0 for d in leaf.by_type)


def test_cost_token_fallback():
    # With no token hooks, content falls back to size // CHARS_PER_TOKEN
    # (the offline / no-tiktoken path); parity with the baseline still holds.
    files = [(BASE / 'src' / 'a.py', 4000)]
    nodes = cost_by_directory(files, BASE, PY_TYPES, MODEL)  # no hooks
    baseline = estimate_cost(
        files, PY_TYPES, MODEL, caching_enabled=False,
    ).baseline_cost_usd
    leaf = nodes['src/a.py']
    assert nodes['.'].total == pytest.approx(baseline)
    assert leaf.docs == len(PY_TYPES)
    # content was 4000 // 4 = 1000 tokens, re-sent per applicable type
    assert leaf.ingestion_cost == pytest.approx(
        (4000 // CHARS_PER_TOKEN) * IN_RATE / 1_000_000 * len(PY_TYPES)
    )


def test_cost_path_outside_base():
    # A path not under base_path keeps its full posix path as the node key
    # rather than crashing.
    nodes = cost_by_directory(
        [(Path('/elsewhere/x.py'), 4000)], BASE, PY_TYPES, MODEL, **_hooks(),
    )
    assert '/elsewhere/x.py' in nodes
    assert nodes['/elsewhere/x.py'].docs == len(PY_TYPES)


def test_format_table():
    tree = [
        (BASE / 'src' / 'a.py', 4000),
        (BASE / 'src' / 'api' / 'b.py', 4000),
        (BASE / 'src' / 'api' / 'c.py', 4000),
    ]
    nodes = cost_by_directory(tree, BASE, PY_TYPES, MODEL, **_hooks())
    leaves = {'src/a.py', 'src/api/b.py', 'src/api/c.py'}

    table = format_by_dir_table(nodes, leaves=leaves, model=MODEL)

    lines = table.splitlines()
    # Header reports the grand total (the root node) and is rate-aware.
    assert f'{nodes["."].total:.2f}' in lines[0]
    # Only directories listed (no file leaves, no root '.' row).
    body = [ln for ln in lines[1:] if ln.strip()]
    assert not any(leaf in line for leaf in leaves for line in body)
    assert len(body) == 2  # exactly src/ and src/api/ — no root, no leaves
    # Directories ranked by total descending: src (incl. api) before src/api.
    src_idx = next(i for i, ln in enumerate(body) if 'src/' in ln and 'api' not in ln)
    api_idx = next(i for i, ln in enumerate(body) if 'src/api' in ln)
    assert src_idx < api_idx
    # Each row shows a dollar figure and a doc count.
    assert '$' in body[src_idx] and 'docs' in body[src_idx]


def test_format_table_empty():
    # No files priced (e.g. nothing to generate) → header with $0.00, no rows.
    table = format_by_dir_table({}, leaves=set(), model=MODEL)
    assert '0.00' in table
    assert table.strip().count('\n') == 0  # header only, no directory rows
