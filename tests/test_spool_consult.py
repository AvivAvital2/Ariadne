"""Consulting spool expert aisles — slices 2-4 of the expert-aisles architecture
(designs/spool-expert-aisles.md §3-4, §7).

Slice 2: load the enabled spools as consultable aisles (themes + endpoint +
taxonomy). Slice 3: route a question to the relevant aisle(s) and consult each
(transport-agnostic via a ``consult`` seam — MCP-backed in production, a fake in
tests). Slice 4: combine the aisle answers with the project context.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from spools import SpoolManifest
from spool_router import Aisle
from spool_consult import (
    AisleAnswer,
    combine,
    consult_relevant,
    load_aisles,
)


class TestLoadAisles:
    def test_builds_aisles_for_enabled_spools(self) -> None:
        themes = {'databricks': ((1.0, 0.0),), 'terraform': ((0.0, 1.0),)}
        tax = {'databricks': ('serialization', 'parallelism'), 'terraform': ()}
        aisles = load_aisles(
            ['databricks', 'terraform'],
            themes_loader=lambda n: themes[n],
            taxonomy_loader=lambda n: tax[n],
            endpoint_for=lambda n: f'mcp://{n}',
        )
        assert [a.name for a in aisles] == ['databricks', 'terraform']
        db = aisles[0]
        assert db.theme_embeddings == ((1.0, 0.0),)
        assert db.taxonomy == ('serialization', 'parallelism')
        assert db.endpoint == 'mcp://databricks'

    def test_skips_aisle_with_no_themes(self) -> None:
        # a spool that produced no themes can't be routed to — drop it rather
        # than register an unroutable aisle.
        aisles = load_aisles(
            ['empty'],
            themes_loader=lambda n: (),
            taxonomy_loader=lambda n: (),
            endpoint_for=lambda n: 'x',
        )
        assert aisles == []


class TestConsultRelevant:
    def _aisles(self):
        return [
            Aisle('databricks', ((1.0, 0.0),), endpoint='mcp://databricks',
                  taxonomy=('serialization',)),
            Aisle('terraform', ((0.0, 1.0),), endpoint='mcp://terraform'),
        ]

    def test_consults_only_routed_aisles(self) -> None:
        consulted = []

        def consult(aisle: Aisle, question: str) -> AisleAnswer:
            consulted.append(aisle.name)
            return AisleAnswer(aisle.name, f'{aisle.name}: {question}')

        answers = consult_relevant(
            'speed this up on databricks', (0.95, 0.05), self._aisles(),
            consult=consult, threshold=0.5,
        )
        assert consulted == ['databricks']          # terraform never consulted
        assert [a.aisle for a in answers] == ['databricks']

    def test_unrelated_question_consults_nothing(self) -> None:
        consulted = []

        def consult(aisle, question):
            consulted.append(aisle.name)
            return AisleAnswer(aisle.name, '')

        answers = consult_relevant(
            'unrelated CSS question', (0.0, 0.0), self._aisles(),
            consult=consult, threshold=0.5,
        )
        assert answers == [] and consulted == []     # no aisle woken


class TestCombine:
    def test_folds_aisle_answers_into_project_context(self) -> None:
        answers = [AisleAnswer(
            'databricks', 'rewrite the loop as rdd.mapPartitions',
            citations=('pyspark.RDD.mapPartitions',))]
        out = combine('demo-spark-proj Analyze loops over bins locally', answers)
        assert 'demo-spark-proj Analyze loops over bins locally' in out   # project kept
        assert 'databricks' in out
        assert 'mapPartitions' in out
        assert 'pyspark.RDD.mapPartitions' in out                 # citation kept

    def test_no_answers_returns_project_context_unchanged(self) -> None:
        out = combine('just my project', [])
        assert 'just my project' in out


class TestAisleTaxonomy:
    """Slice 5: the aisle's advisory lens is declared on the spool (recipe →
    manifest) so a built aisle carries it into consultation."""

    def test_manifest_roundtrips_taxonomy(self) -> None:
        m = SpoolManifest.from_dict({
            'environment': 'databricks', 'version': '1.0.0',
            'target_runtime': 'dbr17.3-lts', 'checksum': 'x',
            'taxonomy': ['serialization', 'parallelism'],
        })
        assert m.taxonomy == ('serialization', 'parallelism')

    def test_manifest_without_taxonomy_defaults_empty(self) -> None:
        m = SpoolManifest.from_dict({
            'environment': 'databricks', 'version': '1.0.0',
            'target_runtime': 'dbr17.3-lts', 'checksum': 'x',
        })
        assert m.taxonomy == ()

    def test_databricks_recipe_declares_taxonomy(self) -> None:
        recipe = yaml.safe_load(
            Path('spool_content/recipes/databricks.yaml').read_text())
        tax = recipe.get('taxonomy') or []
        assert {'parallelism', 'serialization'} <= set(tax)
        # the autolog-patching gotcha is declared, not just generic concerns
        assert 'autolog-patching' in tax
class TestShippedRecipes:
    """Shipped recipes seed `spools create` and round-trip verbatim into the
    working spools.yaml, which is the ONLY file the pack build reads — so a
    fresh create inherits exactly what the shipped recipe declares. These
    pins keep the knowledge fields (and honest naming) in that seed."""

    def test_databricks_recipe_declares_knowledge_fields(self) -> None:
        recipe = yaml.safe_load(
            Path('spool_content/recipes/databricks.yaml').read_text())
        assert 'delta lake' in (recipe.get('name_aliases') or [])
        components = recipe.get('runtime_components') or {}
        assert {'spark', 'delta', 'databricks-sdk-py'} <= set(components)
        assert all(components.values())
        surfaces = recipe.get('surfaces') or {}
        assert {'serialization', 'parallelism', 'io', 'memory',
                'state', 'lifecycle'} <= set(surfaces)
        assert all(isinstance(stems, list) and stems
                   for stems in surfaces.values())

    def test_databricks_recipe_pins_every_corpus_tag(self) -> None:
        # A moving branch (main/master) is not a pin; the blessed versions
        # are known, so the shipped recipe carries real tags.
        recipe = yaml.safe_load(
            Path('spool_content/recipes/databricks.yaml').read_text())
        for repo, spec in recipe['corpus'].items():
            assert spec.get('tag', '').startswith('v'), repo

    def test_opentofu_recipe_replaces_terraform(self) -> None:
        # The artifact is named for its actual corpus: OpenTofu (MPL-2.0)
        # is redistribution-safe; Terraform (BUSL 1.1) is not, and the
        # shipped name must not brand the pack with it.
        recipes = Path('spool_content/recipes')
        assert not (recipes / 'terraform.yaml').exists()
        recipe = yaml.safe_load((recipes / 'opentofu.yaml').read_text())
        assert recipe['name'] == 'opentofu'
        assert recipe['corpus']['opentofu']['url'] == (
            'https://github.com/opentofu/opentofu')
