"""The dry-run generate estimate must price the SAME doc types the
generate phase actually produces.

Bug: the estimate passed ``doc_types=('explanation', 'architecture')``
(2 types) while ``cmd_generate`` defaults to 5
(``explanation, architecture, qa, gotcha, diagram``). Since
``estimate_cost`` sends the file content once per call (per doc type),
estimating 2 types instead of 5 undercounts the generate phase by
~2.5x — a large, systematic under-estimate.
"""
from __future__ import annotations

import argparse

import pytest


@pytest.mark.asyncio
async def test_dry_run_generate_estimate_uses_full_default_doc_types(
    monkeypatch, tmp_path,
):
    from docgen.pricing import CostEstimate

    # Source with one real file so the generate estimate actually runs.
    source_dir = tmp_path / 'src'
    source_dir.mkdir()
    (source_dir / 'mod.py').write_text('def f():\n    return 1\n', encoding='utf-8')

    yaml_path = tmp_path / 'ariadne.yaml'
    yaml_path.write_text(
        f'default_source: src\nsources:\n  src:\n    path: {source_dir}\n',
        encoding='utf-8',
    )
    import config as config_module
    monkeypatch.setattr(
        config_module, '_global_config',
        config_module.Config(config_path=yaml_path),
    )

    # Stub the free phases — we only care about the estimate's inputs.
    monkeypatch.setattr('cli.core.cmd_discover', lambda a: 0)
    monkeypatch.setattr('cli.core.cmd_index', lambda a, **k: 0)

    async def _catalog_sync(a):
        return 0

    monkeypatch.setattr('cli.generation.cmd_catalog_sync', _catalog_sync)

    # Capture the doc_types + caching flag handed to estimate_cost.
    captured: list[tuple] = []
    caching_flags: list = []

    def fake_estimate(**kw):
        captured.append(kw.get('doc_types'))
        caching_flags.append(kw.get('caching_enabled', False))
        return CostEstimate(
            file_count=1, total_calls=1, input_tokens=1, output_tokens=1,
            embedding_tokens=0, llm_cost_usd=0.0, embedding_cost_usd=0.0,
            total_cost_usd=0.0, cost_lower_bound=0.0, cost_upper_bound=0.0,
            model='claude-opus-4-7', rates=(5.0, 25.0),
        )

    monkeypatch.setattr('docgen.pricing.estimate_cost', fake_estimate)

    from cli.generation import cmd_dry_run

    args = argparse.Namespace(
        source='src', model='claude-opus-4-7',
        db=str(tmp_path / 'library.db'), verbose=False, concurrency=None,
    )
    rc = await cmd_dry_run(args)
    assert rc == 0

    assert captured, 'estimate_cost was never called for the generate phase'
    expected = ('explanation', 'architecture', 'qa', 'gotcha', 'diagram')
    # The aggregate generate estimate must price the full default set.
    # (The per-type breakdown additionally calls estimate_cost with
    # single-type tuples — those are expected and fine.)
    assert expected in [tuple(dt) for dt in captured], (
        f'generate aggregate estimate must price all default doc types '
        f'{expected}; captured {[tuple(dt) for dt in captured]}'
    )

    # Anthropic model → caching must be applied (consistent with
    # `generate --dry-run`, and matching the real run which caches the
    # scaffolding). At least one estimate call must enable it.
    assert any(caching_flags), (
        'onboard/dry-run must pass caching_enabled for anthropic models '
        '(consistency with generate --dry-run)'
    )


@pytest.mark.asyncio
async def test_dry_run_generate_estimate_honors_explicit_types(
    monkeypatch, tmp_path,
):
    """When ``--types`` is given (e.g. ``onboard --types
    explanation,architecture``) the estimate must price ONLY those types —
    the same set the real generate phase honors (cli.generate resolves
    ``args.types``). Otherwise the preview over-reports cost AND counts
    staleness against types the run won't produce, making a re-run look like
    it will re-bill everything already generated."""
    from docgen.pricing import CostEstimate

    source_dir = tmp_path / 'src'
    source_dir.mkdir()
    (source_dir / 'mod.py').write_text('def f():\n    return 1\n', encoding='utf-8')

    yaml_path = tmp_path / 'ariadne.yaml'
    yaml_path.write_text(
        f'default_source: src\nsources:\n  src:\n    path: {source_dir}\n',
        encoding='utf-8',
    )
    import config as config_module
    monkeypatch.setattr(
        config_module, '_global_config',
        config_module.Config(config_path=yaml_path),
    )

    monkeypatch.setattr('cli.core.cmd_discover', lambda a: 0)
    monkeypatch.setattr('cli.core.cmd_index', lambda a, **k: 0)

    async def _catalog_sync(a):
        return 0

    monkeypatch.setattr('cli.generation.cmd_catalog_sync', _catalog_sync)

    captured: list[tuple] = []

    def fake_estimate(**kw):
        captured.append(tuple(kw.get('doc_types')))
        return CostEstimate(
            file_count=1, total_calls=1, input_tokens=1, output_tokens=1,
            embedding_tokens=0, llm_cost_usd=0.0, embedding_cost_usd=0.0,
            total_cost_usd=0.0, cost_lower_bound=0.0, cost_upper_bound=0.0,
            model='claude-opus-4-7', rates=(5.0, 25.0),
        )

    monkeypatch.setattr('docgen.pricing.estimate_cost', fake_estimate)

    from cli.generation import cmd_dry_run

    args = argparse.Namespace(
        source='src', model='claude-opus-4-7',
        db=str(tmp_path / 'library.db'), verbose=False, concurrency=None,
        types='explanation,architecture',
    )
    rc = await cmd_dry_run(args)
    assert rc == 0
    assert captured, 'estimate_cost was never called for the generate phase'

    # The aggregate estimate prices exactly the requested two types …
    assert ('explanation', 'architecture') in captured, (
        f'estimate must scope to --types; captured {captured}'
    )
    # … and NOTHING the run won't generate (no qa/gotcha/diagram leaking in).
    leaked = {t for dt in captured for t in dt} & {'qa', 'gotcha', 'diagram'}
    assert not leaked, f'estimate priced un-requested doc types: {leaked}'
