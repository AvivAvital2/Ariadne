"""Render an actual generation prompt and measure its token consumption.

Picks one representative source file from Ariadne, builds the same prompt
the orchestrator would send for ``explanation``, and reports:
  - System prompt token count
  - User template token count (with the file rendered in)
  - Combined total (what Anthropic considers when deciding to cache)
  - The 4096-token threshold for Opus 4.6, with verdict (cache eligible / not)

Run: ``uv run python measure_prompt_tokens.py``
Requires: ANTHROPIC_API_KEY env var.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    # Ariadne uses python-dotenv to load .env at the project root; replicate
    # that here so the script picks up ANTHROPIC_API_KEY the same way the
    # CLI does.
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent / '.env')
    except ImportError:
        pass

    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    if not api_key:
        print(
            'ERROR: ANTHROPIC_API_KEY not set (checked env + .env file)',
            file=sys.stderr,
        )
        return 1

    # Pick a real-world file: medium-size Python module from this codebase.
    target = Path(__file__).parent / 'library' / 'migrations.py'
    if not target.exists():
        print(f'ERROR: target file not found: {target}', file=sys.stderr)
        return 1

    source_code = target.read_text()

    # Reconstruct what the orchestrator passes to render_user_template.
    # The legacy path's module_info comes from format_module_info(...).
    from docgen.prompts import (
        EXPLANATION_PROMPT,
        format_module_info,
        format_related_modules,
        render_user_template,
    )

    module_info = format_module_info(
        module_name='library.migrations',
        docstring=(
            'One-off DB migrations. Hosts migrate_doc_ids — backfill the '
            'deterministic-UUID5 scheme over docs written under the original '
            'UUID4 scheme.'
        ),
        classes=['MigrateDocIdsResult', 'MigrationsMixin'],
        functions=[
            '_compute_primary_key',
            '_delete_doc_with_refs',
            '_rename_doc_refs',
            '_update_staleness_doc_ids',
        ],
    )
    related_modules = format_related_modules([
        ('schema', 'doc_id_for, generate_deterministic_id'),
        ('library', 'Library, _ConnectionProvider'),
        ('library.themes', 'theme_members table'),
    ])

    user_prompt = render_user_template(
        EXPLANATION_PROMPT,
        language='python',
        module_info=module_info,
        source_code=source_code,
        related_modules=related_modules,
    )
    system_prompt = EXPLANATION_PROMPT.system_prompt

    # Display the prompts in full.
    print('=' * 80)
    print('SYSTEM PROMPT')
    print('=' * 80)
    print(system_prompt)
    print()
    print('=' * 80)
    print('USER PROMPT (rendered)')
    print('=' * 80)
    print(user_prompt)
    print()

    # Measure with Anthropic's tokenizer via /messages/count_tokens (httpx,
    # no SDK needed).
    import httpx
    model = 'claude-opus-4-6'

    def count(system, messages) -> int:
        r = httpx.post(
            'https://api.anthropic.com/v1/messages/count_tokens',
            headers={
                'Content-Type': 'application/json',
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01',
                'anthropic-beta': 'prompt-caching-2024-07-31',
            },
            json={'model': model, 'system': system, 'messages': messages},
            timeout=30.0,
        )
        r.raise_for_status()
        return int(r.json()['input_tokens'])

    sys_tokens = count(
        system=system_prompt,
        messages=[{'role': 'user', 'content': 'placeholder'}],
    )
    full_tokens = count(
        system=system_prompt,
        messages=[{'role': 'user', 'content': user_prompt}],
    )
    user_tokens = full_tokens - sys_tokens

    # With cache_control marker — what Anthropic counts toward 4096 minimum.
    cached_full_tokens = count(
        system=[{
            'type': 'text',
            'text': system_prompt,
            'cache_control': {'type': 'ephemeral'},
        }],
        messages=[{'role': 'user', 'content': user_prompt}],
    )

    print('=' * 80)
    print('TOKEN COUNTS (Claude Opus 4.6, 4096-token cache minimum)')
    print('=' * 80)
    print(f'System prompt only:                {sys_tokens:>6,} tokens')
    print(f'User prompt only:                  {user_tokens:>6,} tokens')
    print(f'Combined (system + user):          {full_tokens:>6,} tokens')
    print(f'With cache_control on system:      {cached_full_tokens:>6,} tokens')
    print()
    print(f'File sampled: {target.relative_to(Path(__file__).parent)} '
          f'({len(source_code):,} chars)')
    print()

    # Verdict: would this file's prompt cache?
    threshold = 4096
    print(f'Cache eligibility (Opus 4.6 minimum: {threshold:,} tokens):')
    if sys_tokens >= threshold:
        print(f'  System alone: [PASS] {sys_tokens:,} >= {threshold:,}')
    else:
        print(f'  System alone: [FAIL] {sys_tokens:,} < {threshold:,}')
        print(
            f'    -> system prompt would NOT cache on its own '
            f'({threshold - sys_tokens:,} tokens short)'
        )

    if full_tokens >= threshold:
        print(f'  Full prefix:  [PASS] {full_tokens:,} >= {threshold:,}')
        print(
            '    -> if cache_control were on the FULL user prefix '
            '(per-file payload), it would cache.'
        )
    else:
        print(f'  Full prefix:  [FAIL] {full_tokens:,} < {threshold:,}')
        print(
            f'    -> even system + full user content combined falls short '
            f'by {threshold - full_tokens:,} tokens. This file is too small '
            'to cache regardless of how we structure the prompt.'
        )

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
