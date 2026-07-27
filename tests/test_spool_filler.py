"""Slice (d) of the Spool plugin: the filler-swap generation mode.

The explanation prompt = structural spine + one intent-filler slot
(§18.4). Default mode fills it with raw source (today's behavior,
guarded here); withhold mode renders structure only — no comments, no
docstrings — and admits prose solely through the injectable tier-2
intent_filler. Observable = the BUILT prompt (no LLM call). Synthetic
fixtures only.
"""
import asyncio
import textwrap

from docgen.catalog_enrich import EnrichedElementInfo, EnrichedFileBundle
from docgen.catalog_extractor import ElementInfo
from docgen.generator import DocGenerator, GeneratorConfig


def _bundle(tmp_path):
    module_path = tmp_path / 'fake_module.py'
    module_path.write_text(textwrap.dedent('''
        """MODULE DOCSTRING SECRET."""
        # COMMENT SECRET
        def add(a, b):
            """FUNC DOCSTRING SECRET."""
            return a + b
    ''').strip())
    element = ElementInfo(
        language='python',
        subtype='function',
        file=str(module_path),
        qualified_name='fake_module.add',
        signature='def add(a, b):',
        line_start=3,
        line_end=5,
        col_start=0,
        col_end=16,
    )
    return EnrichedFileBundle(
        path=module_path,
        language='python',
        module_name='fake_module',
        module_docstring='MODULE DOCSTRING SECRET.',
        imports=(),
        elements=(EnrichedElementInfo(element=element),),
        line_count=5,
    )


def _explanation_prompt(generator, bundle):
    prompts = asyncio.run(generator.build_prompts_for_bundle(
        bundle, doc_types=('explanation',),
    ))
    (prompt_bundle,) = prompts
    return prompt_bundle.user_prompt


class TestFillerSwap:
    def test_filler_swap_modes(self, tmp_path):
        bundle = _bundle(tmp_path)

        # Demand 1 — regression guard: the default mode keeps today's
        # behavior — the raw file text (comments included) is in the prompt.
        raw_prompt = _explanation_prompt(DocGenerator(), bundle)
        assert 'COMMENT SECRET' in raw_prompt
        assert 'FUNC DOCSTRING SECRET' in raw_prompt

        # Demand 2 — withhold mode: the structural spine is present, and
        # NO raw source prose survives — not the comment, not the function
        # docstring, not the module docstring (module_info slot included).
        withheld = DocGenerator(
            config=GeneratorConfig(withhold_source_prose=True),
        )
        spine_prompt = _explanation_prompt(withheld, bundle)
        assert 'def add(a, b):' in spine_prompt
        assert 'COMMENT SECRET' not in spine_prompt
        assert 'FUNC DOCSTRING SECRET' not in spine_prompt
        assert 'MODULE DOCSTRING SECRET' not in spine_prompt

        # Demand 3 — the filler: an intent_filler's excerpt is admitted,
        # under the labeled certified-docs (tier 2) section.
        filled = DocGenerator(
            config=GeneratorConfig(withhold_source_prose=True),
            intent_filler=lambda b: 'OFFICIAL EXCERPT ALPHA',
        )
        filled_prompt = _explanation_prompt(filled, bundle)
        assert 'OFFICIAL EXCERPT ALPHA' in filled_prompt
        assert 'certified official documentation' in filled_prompt
        assert 'COMMENT SECRET' not in filled_prompt

    def test_scaladoc_via_scip_stays_out_of_spool_prompts(self, tmp_path):
        # (d+) VERIFICATION demand (§18.6.2 SCIP-transport trap): Scaladoc
        # content arriving via ElementInfo.documentation is tier-3 prose —
        # withhold mode must never render it. Characterizes the property
        # rather than driving new code (structural_source renders only
        # signatures/imports by construction).
        scala_path = tmp_path / 'Engine.scala'
        scala_path.write_text(
            '/** SCALADOC SECRET */\nclass Engine(cores: Int)\n',
        )
        element = ElementInfo(
            language='scala',
            subtype='class',
            file=str(scala_path),
            qualified_name='fake.Engine',
            signature='class Engine(cores: Int)',
            line_start=2,
            line_end=2,
            col_start=0,
            col_end=25,
            documentation={'text': 'SCALADOC SECRET'},
        )
        bundle = EnrichedFileBundle(
            path=scala_path,
            language='scala',
            module_name='fake.Engine',
            imports=(),
            elements=(EnrichedElementInfo(element=element),),
            line_count=2,
        )
        withheld = DocGenerator(
            config=GeneratorConfig(withhold_source_prose=True),
        )
        prompt = _explanation_prompt(withheld, bundle)
        assert 'class Engine(cores: Int)' in prompt
        assert 'SCALADOC SECRET' not in prompt
