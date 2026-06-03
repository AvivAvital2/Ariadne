"""Local token counting for cost estimates via tiktoken.

tiktoken is OpenAI's tokenizer — exact for GPT encodings. For models it
doesn't recognize (newer GPT names, and any non-OpenAI model such as
Claude) we fall back to ``o200k_base``: the right encoding for the
GPT-4o/5 family and a close proxy for Claude (there is no public local
tokenizer for Claude 3+). Counting stays LOCAL, so it scales to
thousands of prompts without a per-prompt API round-trip — unlike
Anthropic's ``count_tokens`` endpoint. And since the dry-run estimate is
presented as a floor with a +50% ceiling, a small cross-tokenizer
discrepancy on Claude is well within tolerance.

Every failure path — tiktoken not installed, encoding unavailable
offline, file unreadable — returns ``None`` so the caller falls back to
its character heuristic instead of crashing a free preview.
"""
from __future__ import annotations

from collections.abc import Callable
from functools import cache
from pathlib import Path


@cache
def _encoding_for(model: str) -> object | None:
    """Return the tiktoken ``Encoding`` for ``model``, or ``None``.

    Tries the model's own encoding first, then ``o200k_base``. Cached so
    the (possibly network-fetched) encoding is built once per model.
    """
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Model not in tiktoken's table (newer GPT names, any Claude
        # model) → o200k_base: right for GPT-4o/5, proxy for Claude.
        try:
            return tiktoken.get_encoding('o200k_base')
        except Exception:  # noqa: BLE001 — offline/download failure → fall back
            return None


def count_text_tokens(text: str, model: str) -> int | None:
    """Token count of ``text`` for ``model``, or ``None`` when no
    tokenizer is available.

    ``disallowed_special=()`` ensures arbitrary source content (which may
    contain literal ``<|...|>`` sequences) is never treated as a special
    token and never raises.
    """
    enc = _encoding_for(model)
    if enc is None:
        return None
    return len(enc.encode(text, disallowed_special=()))


def count_file_tokens(path: Path, model: str) -> int | None:
    """Token count of the file at ``path`` for ``model``. ``None`` if the
    file can't be read or no tokenizer is available.
    """
    try:
        text = Path(path).read_text(encoding='utf-8', errors='ignore')
    except OSError:
        return None
    return count_text_tokens(text, model)


def file_token_counter(model: str) -> Callable[[Path], int | None]:
    """Build a per-path-cached token counter for ``model``.

    Returns ``counter(path) -> int | None`` suitable for
    ``estimate_cost(..., input_tokens_for=...)``: each file is read and
    tokenized at most once, even across the estimate's several passes
    (baseline / batched / per-doc-type). Both dry-run surfaces use this
    one factory so the counting logic isn't duplicated.
    """
    cache: dict[Path, int | None] = {}

    def counter(path: Path) -> int | None:
        if path not in cache:
            cache[path] = count_file_tokens(path, model)
        return cache[path]

    return counter
