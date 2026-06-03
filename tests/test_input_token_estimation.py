"""Input-token estimation via tiktoken (replacing the chars/4 heuristic).

``estimate_cost`` grows an ``input_tokens_for(path)`` hook that
mirrors the existing ``output_tokens_for`` — callers supply exact local
token counts for the file-content (dynamic) portion of the prompt.
``docgen.token_count`` produces those counts with tiktoken: exact for
OpenAI encodings, an ``o200k_base`` proxy for models tiktoken doesn't
recognize (newer GPT names and Claude). Every failure path degrades to
``None`` so the caller falls back to the character heuristic and a free
offline dry-run never crashes.
"""
from __future__ import annotations

from docgen.pricing import CHARS_PER_TOKEN, estimate_cost

# ---------------------------------------------------------------------------
# estimate_cost: input_tokens_for hook
# ---------------------------------------------------------------------------


def test_input_tokens_for_overrides_char_heuristic(tmp_path):
    f = tmp_path / 'a.py'
    f.write_text('x' * 4000)  # size 4000 → chars/4 = 1000 dynamic tokens
    files = [(f, f.stat().st_size)]

    base = estimate_cost(files, ('explanation',), 'gpt-5.4')
    hooked = estimate_cost(
        files, ('explanation',), 'gpt-5.4',
        input_tokens_for=lambda _path: 250,
    )
    # One call for a .py file; only the dynamic (content) term changes,
    # from 1000 (chars/4) to the hook's 250.
    assert base.input_tokens - hooked.input_tokens == (
        int(4000 // CHARS_PER_TOKEN) - 250
    )


def test_input_tokens_for_none_falls_back_to_chars(tmp_path):
    f = tmp_path / 'a.py'
    f.write_text('x' * 4000)
    files = [(f, f.stat().st_size)]

    base = estimate_cost(files, ('explanation',), 'gpt-5.4')
    fell_back = estimate_cost(
        files, ('explanation',), 'gpt-5.4',
        input_tokens_for=lambda _path: None,
    )
    assert fell_back.input_tokens == base.input_tokens


def test_file_token_counter_is_shared_and_cached(tmp_path):
    """Both dry-run surfaces use this one factory; it reads+tokenizes
    each path at most once.
    """
    from docgen import token_count

    f = tmp_path / 'snippet.py'
    f.write_text('def add(a, b):\n    return a + b\n')

    calls = {'n': 0}
    real = token_count.count_file_tokens

    def counting(path, model):
        calls['n'] += 1
        return real(path, model)

    token_count.count_file_tokens = counting
    try:
        counter = token_count.file_token_counter('gpt-5.4')
        first = counter(f)
        second = counter(f)
    finally:
        token_count.count_file_tokens = real

    assert first == second
    assert calls['n'] == 1, 'second lookup must hit the cache'


# ---------------------------------------------------------------------------
# docgen.token_count: tiktoken-backed counts with graceful fallback
# ---------------------------------------------------------------------------


def test_count_text_tokens_real_or_none():
    """With tiktoken available, a positive count; without it (or its
    encoding) a ``None``. Portable across both environments.
    """
    from docgen.token_count import count_text_tokens

    n = count_text_tokens('hello world, this is a short test.', 'gpt-5.4')
    assert n is None or (isinstance(n, int) and n > 0)


def test_count_file_tokens_reads_file(tmp_path):
    from docgen.token_count import count_file_tokens

    f = tmp_path / 'snippet.py'
    f.write_text('def add(a, b):\n    return a + b\n')
    n = count_file_tokens(f, 'gpt-5.4')
    assert n is None or (isinstance(n, int) and n > 0)


def test_count_file_tokens_missing_file_is_none(tmp_path):
    from docgen.token_count import count_file_tokens

    assert count_file_tokens(tmp_path / 'does-not-exist.py', 'gpt-5.4') is None


def test_count_text_tokens_graceful_without_tiktoken(monkeypatch):
    """Force the tiktoken-import-failure path so the helper degrades to
    ``None`` regardless of whether tiktoken is installed in this env.
    """
    import builtins

    from docgen import token_count

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'tiktoken':
            raise ImportError('simulated missing tiktoken')
        return real_import(name, *args, **kwargs)

    token_count._encoding_for.cache_clear()
    monkeypatch.setattr(builtins, '__import__', fake_import)
    try:
        assert token_count.count_text_tokens('hello', 'gpt-5.4') is None
    finally:
        token_count._encoding_for.cache_clear()
