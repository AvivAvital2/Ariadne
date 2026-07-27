"""Contract test for the ``ariadne_estimate`` MCP tool — the structured
cost preview behind the onboarding "Preview" step (per-directory explorer,
doc-type matrix, model picker).

A single evolving test, one demand at a time:

  D1 — estimating a source returns positive totals, the right file count,
       and a language histogram matching the synthetic tree.
  D2 — the per-doc-type and per-directory breakdowns are populated and
       reconcile with the file set (every directory the files live in
       appears; doc-type rows cover the requested types).
  D3 — the model picker data is present: every LLM_PRICING model with its
       rates, and the per-language applicable doc types.
  D4 — batching never costs more than live, and is strictly cheaper when
       there is any LLM work to do.
  D6 — each configured exclusion reports how much it removes from scope:
       an exclude glob saves a positive amount matching its file count, a
       glob that matches nothing saves $0, a force-included (exempt) dir
       shows a NEGATIVE saving (it adds cost), and the reported glob saving
       reconciles with the real total delta of applying it.

No LLM calls — ``estimate_*`` are pure functions over the walked files.
Costs depend on token heuristics, so assertions check structure/ordering,
not exact dollar amounts.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ariadne_mcp.server_admin import ariadne_estimate, ariadne_source_add


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch, tmp_path):
    """Isolated, pre-created ariadne.yaml (so ARIADNE_CONFIG is authoritative
    rather than shadowed by the package-root config) + reset singletons."""
    cfg_path = tmp_path / 'ariadne.yaml'
    cfg_path.write_text('sources: {}\n')
    monkeypatch.setenv('ARIADNE_CONFIG', str(cfg_path))
    monkeypatch.chdir(tmp_path)

    import config as config_module
    monkeypatch.setattr(config_module, '_global_config', None, raising=False)

    from ariadne_mcp.service import AriadneService
    monkeypatch.setattr(AriadneService, '_instance', None, raising=False)
    return cfg_path


def _make_source(root: Path) -> None:
    """A small synthetic tree: 2 python + 1 javascript + 1 markdown + 1 Dockerfile."""
    (root / 'pkg').mkdir(parents=True)
    (root / 'pkg' / 'a.py').write_text('def a():\n    return 1\n')
    (root / 'pkg' / 'b.py').write_text('def b():\n    return 2\n')
    (root / 'web').mkdir()
    (root / 'web' / 'app.js').write_text('export const f = () => 3;\n')
    (root / 'README.md').write_text('# Project\n\nDocs.\n')
    (root / 'Dockerfile').write_text('FROM python:3.12\nENV X=1\nEXPOSE 8080\n')


async def test_estimate_tool_evolves_through_contract(monkeypatch, tmp_path):
    src = tmp_path / 'proj'
    src.mkdir()
    _make_source(src)
    await ariadne_source_add('proj', path=str(src))

    # ---- D1: totals, file count, language histogram ------------------
    est = await ariadne_estimate('proj', model='claude-opus-4-8')
    assert est.source == 'proj'
    assert est.model == 'claude-opus-4-8'
    assert est.file_count == 5
    assert est.total_cost_usd > 0
    assert est.cost_lower_bound <= est.total_cost_usd <= est.cost_upper_bound

    langs = {lc.language: lc.files for lc in est.languages}
    assert langs.get('python') == 2
    assert sum(lc.files for lc in est.languages) == 5
    # Percentages are shares of the file set.
    assert abs(sum(lc.percent for lc in est.languages) - 100.0) < 1.0

    # ---- D2: per-doc-type and per-directory breakdowns ---------------
    types = {d.doc_type for d in est.by_doc_type}
    assert 'explanation' in types
    assert all(d.cost_usd >= 0 for d in est.by_doc_type)

    dirs = {d.rel_path for d in est.by_directory}
    # Every directory the files live in is represented (root, pkg, web).
    assert 'pkg' in dirs and 'web' in dirs
    assert sum(d.docs for d in est.by_directory if d.rel_path in ('pkg', 'web')) > 0

    # ---- D3: model picker + per-language doc types -------------------
    models = {m.model: (m.input_per_million, m.output_per_million)
              for m in est.available_models}
    assert models.get('claude-opus-4-8') == (5.0, 25.0)
    # The picker is derived from LLM_PRICING and offers the generally-available
    # models but NOT invitation-only ones: claude-mythos-5 (Project Glasswing)
    # is excluded, while claude-fable-5 and claude-sonnet-5 remain.
    assert 'claude-mythos-5' not in models
    assert 'claude-fable-5' in models
    assert 'claude-sonnet-5' in models
    assert 'explanation' in est.language_doc_types.get('python', [])

    # ---- D4: batch is never costlier, and cheaper when work exists ---
    assert est.total_cost_batched_usd <= est.total_cost_usd
    assert est.total_cost_batched_usd < est.total_cost_usd

    # ---- D5: Dockerfiles are accounted for (catalog/structured, and
    #         explanation-only via the cost model's default) -----------
    assert langs.get('dockerfile') == 1
    assert est.language_doc_types.get('dockerfile') == ['explanation']

    # ---- D6: per-exclusion savings ----------------------------------
    # A second source whose tree has rst docs (a glob target) and a
    # default-excluded ``vendor`` dir the user force-includes (an exempt).
    src2 = tmp_path / 'proj2'
    (src2 / 'pkg').mkdir(parents=True)
    (src2 / 'pkg' / 'a.py').write_text('def a():\n    return 1\n')
    (src2 / 'pkg' / 'b.py').write_text('def b():\n    return 2\n')
    (src2 / 'docs' / 'guide').mkdir(parents=True)
    (src2 / 'docs' / 'intro.rst').write_text('Intro\n=====\n\nText body.\n')
    (src2 / 'docs' / 'guide' / 'usage.rst').write_text('Usage\n=====\n\nMore body.\n')
    (src2 / 'vendor').mkdir()
    (src2 / 'vendor' / 'lib.py').write_text('def v():\n    return 9\n')

    # Baseline: force-include vendor, no glob excludes yet → 5 files
    # (pkg/a, pkg/b, vendor/lib, 2× rst).
    await ariadne_source_add('proj2', path=str(src2), exempt_dirs=['vendor'])
    base = await ariadne_estimate('proj2', model='claude-opus-4-8')
    assert base.file_count == 5

    # Exclude rst (+ a glob that matches nothing); vendor stays exempt.
    await ariadne_source_add('proj2', exclude=['**/*.rst', '**/*.zzz'])
    est2 = await ariadne_estimate('proj2', model='claude-opus-4-8')
    assert est2.file_count == 3          # rst gone; vendor still force-included

    saved = {s.pattern: s for s in est2.exclusion_savings}
    assert set(saved) == {'**/*.rst', '**/*.zzz', 'vendor'}

    rst = saved['**/*.rst']
    assert rst.kind == 'glob' and rst.files == 2 and rst.saved_usd > 0
    assert 0 <= rst.saved_batched_usd <= rst.saved_usd

    # A glob that matches nothing reports a real zero — not just an echo.
    zzz = saved['**/*.zzz']
    assert zzz.kind == 'glob' and zzz.files == 0 and zzz.saved_usd == 0

    # Force-including a default-skipped dir ADDS cost → negative saving.
    ven = saved['vendor']
    assert ven.kind == 'exempt' and ven.files == 1 and ven.saved_usd < 0

    # The reported glob saving reconciles with the real total delta:
    # re-including the rst would raise the total by ~that amount.
    assert base.total_cost_usd > est2.total_cost_usd
    assert rst.saved_usd == pytest.approx(
        base.total_cost_usd - est2.total_cost_usd, rel=0.25)
