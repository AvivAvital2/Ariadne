"""Cost estimation for ``ariadne generate --dry-run``.

Estimates the LLM cost of a generation run from the file list, the
requested doc types, and the model name. No LLM calls — pure math
over file sizes and a static rate table.

The estimate is character-based (≈4 chars/token); accuracy is ±50%.
The returned ``CostEstimate`` carries a lower/upper bound to make
the uncertainty visible to users.

Update ``LLM_PRICING`` when API rates change. Models not in the table
return an estimate with ``rates=None`` and ``total_cost_usd=0`` so the
caller can prompt the user to add the model rather than silently
producing a misleading number.
"""
from __future__ import annotations

from pathlib import Path

from attrs import frozen

# Per-million-token prices in USD: (input, output).
# Verified against published rates as of 2026-04. Bump when prices change.
LLM_PRICING: dict[str, tuple[float, float]] = {
    'gpt-5.2': (1.75, 14.00),
    'gpt-5.4': (2.50, 15.00),
    'gpt-5.5': (5.00, 30.00),
    'claude-sonnet-4-6': (3.00, 15.00),
    'claude-opus-4-6': (5.00, 25.00),
    'claude-opus-4-7': (5.00, 25.00),
    'claude-opus-4-8': (5.00, 25.00),
}

# text-embedding-3-large per-million-token price.
EMBEDDING_PRICING_PER_M: float = 0.13

# ~4 chars per token for English code is a reasonable midpoint;
# subclasses of this constant let us widen the uncertainty band.
CHARS_PER_TOKEN: float = 4.0

# Per-call prompt scaffolding (system prompt + format spec + framing).
# Conservative estimate; some prompts are ~2k tokens.
PROMPT_OVERHEAD_TOKENS: int = 800

# Average output length per generated doc; varies hugely between explanation
# (long) and diagram (short). Use the mid-range explanation/architecture as
# the default since that's the dominant usage.
AVG_OUTPUT_TOKENS_PER_CALL: int = 1500

# Rough embedding tokens generated per LLM-produced doc (chunks + sections
# + doc-level). Pulled from observed run data.
EMBEDDING_TOKENS_PER_CALL: int = 5000


@frozen
class CostEstimate:
    """Result of ``estimate_cost``.

    ``llm_cost_usd`` / ``total_cost_usd`` reflect ``caching_enabled`` and
    ``batch_discount`` if those were passed; otherwise they're the raw
    rate-card numbers. ``baseline_cost_usd`` always reflects the no-cache,
    no-batch reference so the savings are explicit to the caller.
    """
    file_count: int
    total_calls: int
    input_tokens: int
    output_tokens: int
    embedding_tokens: int
    llm_cost_usd: float
    embedding_cost_usd: float
    total_cost_usd: float
    cost_lower_bound: float
    cost_upper_bound: float
    model: str
    rates: tuple[float, float] | None  # (input_per_M, output_per_M)
    baseline_cost_usd: float = 0.0
    cache_savings_usd: float = 0.0
    batch_savings_usd: float = 0.0


# Anthropic prompt-caching multipliers. Cache writes cost 25% MORE than
# base; cache reads cost 10% of base. The crossover is at ~2 reuses, so
# any doc-type called for ≥3 files comes out ahead.
_CACHE_WRITE_MULT: float = 1.25
_CACHE_READ_MULT: float = 0.10

# Anthropic Message Batches discount: 50% off both input and output.
_BATCH_DISCOUNT: float = 0.50


def _detect_language(path: Path) -> str | None:
    """Mirror of catalog_extractor._detect_language for the languages we
    can estimate. Kept inline to avoid importing the heavy extractor
    just to read a file extension.
    """
    ext = path.suffix.lower()
    if ext == '.py':
        return 'python'
    if ext in ('.html', '.htm'):
        return 'html'
    if ext in ('.js', '.jsx', '.ts', '.tsx', '.mjs'):
        return 'javascript'
    if ext == '.json':
        return 'json'
    if ext in ('.yaml', '.yml'):
        return 'yaml'
    if ext in ('.md', '.markdown'):
        return 'markdown'
    if ext in ('.scala', '.sbt'):
        return 'scala'
    if ext == '.java':
        return 'java'
    return None


def _supported_doc_types_for(language: str | None) -> tuple[str, ...]:
    """Per-language supported doc types — mirrors prompts.LANGUAGE_DOC_TYPES.

    Imported inline to avoid circular imports through prompts → catalog stack.
    """
    from docgen.prompts import LANGUAGE_DOC_TYPES
    if language is None:
        return ('explanation', 'architecture', 'qa', 'catalog', 'gotcha', 'diagram')
    return LANGUAGE_DOC_TYPES.get(language, ('explanation',))


def estimate_cost(
    files: tuple[tuple[Path, int], ...] | list[tuple[Path, int]],
    doc_types: tuple[str, ...],
    model: str,
    *,
    caching_enabled: bool = False,
    batch_enabled: bool = False,
    output_tokens_for=None,
) -> CostEstimate:
    """Estimate total cost for a generation run.

    ``output_tokens_for(doc_type, language) -> float | None`` (optional)
    supplies the *calibrated* average output tokens per call for a bucket
    (from real past-run usage). When it returns a value, it replaces the
    flat ``AVG_OUTPUT_TOKENS_PER_CALL`` heuristic for that call;
    ``None`` falls back to the heuristic. This is what makes the estimate
    self-tune to the codebase + model after a run.

    Args:
        files: Iterable of ``(path, size_in_bytes)`` for every file the
            run will process. ``size_in_bytes`` controls input-token
            estimation.
        doc_types: Doc types requested by the user (CLI ``--types``);
            filtered per-language by ``LANGUAGE_DOC_TYPES``.
        model: Model name as configured (e.g. ``"gpt-5.4"`` or
            ``"claude-opus-4-6"``).
        caching_enabled: Apply Anthropic prompt-caching savings on the
            static prompt scaffolding (``PROMPT_OVERHEAD_TOKENS``).
            Per doc-type, the first call writes at 1.25× and subsequent
            calls read at 0.10× — large savings for runs touching many
            files per doc-type. Pass True when the provider is anthropic
            and ``cache_system_prompt`` will be on.
        batch_enabled: Apply Anthropic's Message Batches 50%-off discount
            to the post-cache LLM cost. Pass True when ``--batch`` is set
            (or auto-batch will trigger).

    Returns:
        A ``CostEstimate`` with token counts, dollar cost, and an
        uncertainty band. ``rates is None`` indicates the model isn't
        in ``LLM_PRICING``; cost will be zero so the caller can warn.
        ``baseline_cost_usd`` is always the unoptimized reference;
        ``cache_savings_usd`` and ``batch_savings_usd`` break down the
        discounts that produced ``total_cost_usd``.
    """
    files_list = list(files)
    rates = LLM_PRICING.get(model)

    # Track per-doc-type call counts so cache reuse can be priced per type.
    calls_per_type: dict[str, int] = {}
    dynamic_input_tokens = 0  # file-content portion (no caching applies)
    output_tokens = 0
    total_calls = 0

    for path, size in files_list:
        lang = _detect_language(path)
        supported = _supported_doc_types_for(lang)
        effective = tuple(t for t in doc_types if t in supported)
        calls = len(effective)
        if calls == 0:
            continue
        per_call_dynamic = size // CHARS_PER_TOKEN
        dynamic_input_tokens += int(per_call_dynamic * calls)
        total_calls += calls
        for t in effective:
            calls_per_type[t] = calls_per_type.get(t, 0) + 1
            # Output: calibrated per-bucket average when available, else
            # the flat heuristic. Output is ~independent of file size, so
            # a per-(doc_type, language) average is a strong signal.
            out = None
            if output_tokens_for is not None:
                out = output_tokens_for(t, lang)
            output_tokens += int(
                out if out is not None else AVG_OUTPUT_TOKENS_PER_CALL
            )

    static_input_uncached = PROMPT_OVERHEAD_TOKENS * total_calls
    input_tokens = dynamic_input_tokens + static_input_uncached
    embedding_tokens = total_calls * EMBEDDING_TOKENS_PER_CALL

    if rates is None:
        return CostEstimate(
            file_count=len(files_list),
            total_calls=total_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            embedding_tokens=embedding_tokens,
            llm_cost_usd=0.0,
            embedding_cost_usd=0.0,
            total_cost_usd=0.0,
            cost_lower_bound=0.0,
            cost_upper_bound=0.0,
            model=model,
            rates=None,
        )

    in_rate, out_rate = rates

    # Baseline LLM cost (no caching, no batch) for comparison. Embedding is
    # tracked separately so the savings rows below stay apples-to-apples.
    baseline_llm = (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000
    embedding_cost = (embedding_tokens * EMBEDDING_PRICING_PER_M) / 1_000_000

    # Apply caching: each doc-type's static prefix is paid once at 1.25× then
    # 0.1× per subsequent call. A doc-type with n calls contributes
    # PROMPT_OVERHEAD_TOKENS × (1.25 + 0.1 × (n - 1)) instead of × n.
    if caching_enabled:
        static_input_cached = sum(
            PROMPT_OVERHEAD_TOKENS * (_CACHE_WRITE_MULT + _CACHE_READ_MULT * (n - 1))
            for n in calls_per_type.values() if n > 0
        )
        effective_input = dynamic_input_tokens + static_input_cached
    else:
        effective_input = input_tokens

    llm_cost = (effective_input * in_rate + output_tokens * out_rate) / 1_000_000
    cache_savings = max(0.0, baseline_llm - llm_cost)

    # Apply batch discount on the post-cache LLM cost. Embeddings aren't batched.
    if batch_enabled:
        batch_savings = llm_cost * _BATCH_DISCOUNT
        llm_cost -= batch_savings
    else:
        batch_savings = 0.0

    total = llm_cost + embedding_cost

    # Uncertainty band: ±50% reflects the character-based heuristic.
    return CostEstimate(
        file_count=len(files_list),
        total_calls=total_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        embedding_tokens=embedding_tokens,
        llm_cost_usd=llm_cost,
        embedding_cost_usd=embedding_cost,
        total_cost_usd=total,
        cost_lower_bound=total * 0.5,
        cost_upper_bound=total * 1.5,
        model=model,
        rates=rates,
        baseline_cost_usd=baseline_llm,
        cache_savings_usd=cache_savings,
        batch_savings_usd=batch_savings,
    )


def per_doc_generation_cost(model: str) -> float:
    """Rough LLM cost of generating one doc, for the ROI calculation.

    Tracks the configured model's rate card (one LLM call ≈ a typical
    file's content + scaffolding in, one doc body out) instead of a
    fixed constant. Falls back to a Sonnet-ish midpoint for models not
    in ``LLM_PRICING`` so ROI never silently zeroes out.
    """
    in_rate, out_rate = LLM_PRICING.get(model, (3.0, 15.0))
    # Representative single-call token counts (scaffolding + a ~6 KB
    # file's worth of input, one doc body out).
    input_tokens = PROMPT_OVERHEAD_TOKENS + 1500
    output_tokens = AVG_OUTPUT_TOKENS_PER_CALL
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


def estimate_generate_by_doc_type(
    files,
    doc_types: tuple[str, ...],
    model: str,
    *,
    caching_enabled: bool = False,
    batch_enabled: bool = False,
    output_tokens_for=None,
) -> list[tuple[str, CostEstimate]]:
    """Per-doc-type cost breakdown for the generate phase.

    Returns ``[(doc_type, CostEstimate), ...]`` — one estimate per
    requested type — so callers can show what each type contributes and
    let the user decide which to include. With caching off (the default
    for the dry-run), the per-type costs sum to the aggregate
    :func:`estimate_cost` over all ``doc_types``.
    """
    files_list = list(files)
    return [
        (
            t,
            estimate_cost(
                files=files_list, doc_types=(t,), model=model,
                caching_enabled=caching_enabled,
                batch_enabled=batch_enabled,
                output_tokens_for=output_tokens_for,
            ),
        )
        for t in doc_types
    ]
