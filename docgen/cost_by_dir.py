"""Tier 2 of the dry-run explorer: decompose the generate cost per directory.

The dry-run's headline *generate* number is
``pricing.estimate_cost(gen_files, doc_types, model)`` — per file, one LLM
call per *applicable* doc type (language curation), file content re-sent per
type, output tokens calibrated. This module decomposes that estimate per
directory and file so the explorer (Tier 3) can show the real cost of each
subtree and a ``ariadne dry-run --by-dir`` table can print it.

Parity is exact because ``estimate_cost``'s ``baseline_cost_usd`` (no
caching, no batch) is **linear per file**: the per-file/per-doc-type pieces
sum to the aggregate. We therefore decompose against the **baseline** and
leave caching / batch / embedding savings as global footers (they are
cross-file non-linear or differently billed). The per-type pieces come from
``estimate_generate_by_doc_type`` so there is no parallel pricing path — the
per-dir numbers ARE the dry-run's numbers.

``gen_cost`` is the scaffold-input + output cost of a doc type; the file
content's input cost is peeled out into ``ingestion_cost`` (a single line)
so a big-content / few-docs file — e.g. a vendored bootstrap installer —
reads as the cost hog it is.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path, PurePosixPath

from attrs import frozen

from docgen.pricing import (
    CHARS_PER_TOKEN,
    LLM_PRICING,
    estimate_generate_by_doc_type,
)


@frozen
class DocTypeCost:
    """Per-doc-type cost at a node. ``gen_cost`` excludes content ingestion
    (that is summed separately into ``NodeCost.ingestion_cost``)."""

    doc_type: str
    count: int            # docs of this type generated in the subtree
    gen_cost: float       # scaffold-input + output $ (baseline)


@frozen
class NodeCost:
    """Decomposed generate cost for one directory or file (by ``rel_path``)."""

    rel_path: str
    by_type: tuple[DocTypeCost, ...]
    ingestion_cost: float     # content_tokens × (#applicable types) × in_rate
    total: float              # ingestion + Σ by_type.gen_cost (== baseline LLM $)
    docs: int                 # Σ by_type.count


def cost_by_directory(
    files,
    base_path,
    doc_types: tuple[str, ...],
    model: str,
    *,
    output_tokens_for=None,
    input_tokens_for=None,
    prompt_overhead_for=None,
    estimator: Callable = estimate_generate_by_doc_type,
) -> dict[str, NodeCost]:
    """Decompose the generate estimate over ``files`` into per-node costs.

    ``files`` is the dry-run's ``gen_files`` (``[(Path, size_bytes)]``);
    ``base_path`` rel-ifies each path to its directory key. The three hooks
    are the dry-run's calibration hooks, passed straight through to the
    estimator. Returns one :class:`NodeCost` per directory and per file
    rel_path (Tier 3 overlays these on Tier 1's ``ScanNode``s).
    """
    base = Path(base_path)
    rates = LLM_PRICING.get(model)
    in_rate = rates[0] if rates else 0.0  # no rate card → cost 0, like estimate_cost

    # rel_path -> {by_type: {doc_type: [count, gen_cost]}, ingestion, total, docs}
    accs: dict = {}

    def _acc(key):
        a = accs.get(key)
        if a is None:
            a = {'by_type': {}, 'ingestion': 0.0, 'total': 0.0, 'docs': 0}
            accs[key] = a
        return a

    for path, size in files:
        rel = _rel(path, base)
        per_type = estimator(
            [(path, size)], doc_types, model,
            caching_enabled=False,  # baseline is linear per file ⇒ additive
            output_tokens_for=output_tokens_for,
            input_tokens_for=input_tokens_for,
            prompt_overhead_for=prompt_overhead_for,
        )

        # Content tokens, peeled from each applicable type's cost as the
        # ingestion line — mirror estimate_cost's own input resolution so
        # the peel is exact (content is re-sent once per doc type).
        content = input_tokens_for(path) if input_tokens_for is not None else None
        if content is None:
            content = int(size // CHARS_PER_TOKEN)
        ingestion_per_type = content * in_rate / 1_000_000

        file_by_type: list[tuple[str, float]] = []
        file_ingestion = 0.0
        file_total = 0.0
        for doc_type, est in per_type:
            if est.total_calls == 0:
                continue  # doc type not applicable to this file's language
            file_by_type.append((doc_type, est.baseline_cost_usd - ingestion_per_type))
            file_ingestion += ingestion_per_type
            file_total += est.baseline_cost_usd

        # Credit the file leaf and every ancestor directory ('.' = root).
        ancestors = [p.as_posix() for p in PurePosixPath(rel).parents]
        for key in (rel, *ancestors):
            a = _acc(key)
            for doc_type, gen_cost in file_by_type:
                slot = a['by_type'].setdefault(doc_type, [0, 0.0])
                slot[0] += 1
                slot[1] += gen_cost
            a['ingestion'] += file_ingestion
            a['total'] += file_total
            a['docs'] += len(file_by_type)

    return {
        key: NodeCost(
            rel_path=key,
            by_type=tuple(
                DocTypeCost(doc_type=t, count=c, gen_cost=cost)
                for t, (c, cost) in sorted(a['by_type'].items())
            ),
            ingestion_cost=a['ingestion'],
            total=a['total'],
            docs=a['docs'],
        )
        for key, a in accs.items()
    }


def format_by_dir_table(nodes, *, leaves, model: str, top: int = 20) -> str:
    """Render a static per-directory cost table (the non-TTY ``--by-dir``
    surface Tier 3 falls back to).

    ``nodes`` is :func:`cost_by_directory`'s output; ``leaves`` is the set of
    file rel_paths (so only *directory* rollups are listed). Directories are
    ranked by ``total`` descending and capped at ``top``; the grand total
    (the root ``'.'`` node) is shown in the header. The figure is a
    token-value estimate at the model's rate card, **baseline (pre-caching)**
    — the dry-run headline applies caching on top, so this sums slightly
    higher than the cached number.
    """
    root_total = nodes['.'].total if '.' in nodes else 0.0
    header = (
        f'Per-directory generate cost — ${root_total:.2f} token-value '
        f'({model}, baseline / pre-caching):'
    )
    dirs = sorted(
        (n for key, n in nodes.items() if key != '.' and key not in leaves),
        key=lambda n: n.total,
        reverse=True,
    )
    rows = [
        f'  ${n.total:>8.2f}  {n.rel_path + "/":<28} {n.docs} docs'
        for n in dirs[:top]
    ]
    return '\n'.join([header, *rows])


def _rel(path, base) -> str:
    """``path`` relative to ``base`` as a posix string (best-effort)."""
    path = Path(path)
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()
