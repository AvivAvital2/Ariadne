"""Evolutionary-TDD walk for ``.rst`` catalog coverage —
``designs/rst-support.md``.

reStructuredText is human-authored (Sphinx) documentation for Python code.
Tier 1 makes it a recognized, discovered, correctly-priced language; Tier 2
extracts its sections (via the docutils doctree) as ``rst_section`` catalog
elements with correct nesting; Tier 3 captures Sphinx ``autodoc`` directives
that reference code symbols.

The file grows one demand at a time across the tiers' slices.
"""
from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import get_args

import numpy as np

from config import get_config
from docgen import catalog_enrich, catalog_extractor, pricing, scip_persist
from docgen.catalog_extractor import (
    ElementInfo,
    Language,
    Subtype,
    _detect_language,
    extract_elements,
)
from docgen.catalog_writer import CATALOG_EXTS, sync_file_catalog
from docgen.generator import DocGenerator, GeneratorConfig
from docgen.prompts import (
    ARCHITECTURE_PROMPT,
    EXPLANATION_PROMPT,
    LANGUAGE_DOC_TYPES,
    filter_doc_types_for_language,
    format_rst_crossref,
    render_user_template,
)
from docgen.scip_cross_source import CrossSourceGraph
from docgen.scip_extractor import ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol
from docgen.staleness import find_catalog_files
from library import Library, scip
from scope_resolution import make_scoped_library
from tests._scoped_config_fixture import install_test_config
from tests.test_persist_all_sources import (
    _synthetic_python_index_with_one_class,
    _write_manifest,
)
from writer import LibraryWriter

# rst is documentation prose: it earns explanation + architecture + qa +
# gotcha, but not catalog/diagram (no symbol structure to catalog/diagram).
RST_DOC_TYPES = ('explanation', 'architecture', 'qa', 'gotcha')
# The full doc-type set a caller might request (Python/Scala get all six).
ALL_DOC_TYPES = ('explanation', 'architecture', 'qa', 'catalog', 'gotcha', 'diagram')


class TestRstCoverage:
    # ---- Slice 1.1 — detection + type honesty ------------------------
    # Smallest demand: a ``.rst`` path resolves to the ``'rst'`` language.
    # Until this holds, ``.rst`` falls through to ``None`` and is never
    # discovered, priced, or documented (the reported ``$0.00`` bug).
    def test_detect_language_returns_rst(self) -> None:
        assert _detect_language(Path('docs/guide.rst')) == 'rst'
        # The new `.rst` branch must not swallow other suffixes — an
        # unrecognized extension still falls through to None (also covers
        # the False arm of the new branch).
        assert _detect_language(Path('notes.txt')) is None

    # ``Language`` (the Literal annotating ``ElementInfo.language`` etc.)
    # must include ``'rst'`` so the new value is statically valid.
    def test_rst_in_language_literal(self) -> None:
        assert 'rst' in get_args(Language)

    # ---- Slice 1.2 — discovery ---------------------------------------
    # ``.rst`` must be in ``CATALOG_EXTS`` so the discovery walk visits it;
    # without this the file is dropped before pricing (the ``$0.00`` bug).
    def test_rst_in_catalog_exts(self) -> None:
        assert '.rst' in CATALOG_EXTS

    # A synthetic ``.rst`` is collected by the catalog walk. (Tier 1 made it
    # file-index-only; Tier 2 adds section extraction — see the Tier 2
    # tests below for the extraction contract.)
    def test_find_catalog_files_collects_rst(self, tmp_path: Path) -> None:
        rst = tmp_path / 'guide.rst'
        rst.write_text(
            'Widget Toolkit\n==============\n\nOverview of the toolkit.\n',
            encoding='utf-8',
        )
        assert rst in find_catalog_files(tmp_path)

    # ---- Slice 1.3 — correct cost + curation (the $0 fix) ------------
    # rst gets a documentation doc-type set in BOTH the generate path
    # (filter_doc_types_for_language) and as the LANGUAGE_DOC_TYPES entry.
    def test_rst_doc_types_are_documentation_set(self) -> None:
        assert LANGUAGE_DOC_TYPES.get('rst') == RST_DOC_TYPES
        assert filter_doc_types_for_language(ALL_DOC_TYPES, 'rst') == RST_DOC_TYPES

    # The pricing mirror must resolve ``.rst`` to 'rst' (matching the
    # catalog detector), so cost curation uses rst's set — not the
    # over-counting None->full-set fallback.
    def test_rst_priced_and_consistent_across_detectors(self) -> None:
        assert pricing._detect_language(Path('x.rst')) == 'rst'
        assert _detect_language(Path('x.rst')) == pricing._detect_language(Path('x.rst'))
        assert pricing._supported_doc_types_for('rst') == RST_DOC_TYPES
        # cover the False arm of the new branch in the pricing mirror too
        assert pricing._detect_language(Path('notes.txt')) is None

    # The reported ``$0.00`` is fixed: a discovered .rst prices > 0, curated
    # to exactly its four doc types (one LLM call each) — not the full six.
    def test_estimate_cost_prices_rst_nonzero_at_four_types(self) -> None:
        model = next(iter(pricing.LLM_PRICING))
        est = pricing.estimate_cost([(Path('x.rst'), 4000)], ALL_DOC_TYPES, model)
        assert est.total_calls == len(RST_DOC_TYPES)  # 4 applicable types
        assert est.total_cost_usd > 0

    # ---- Slice 1.4 — file-level doc e2e (design-intent) -------------
    # Composition check: catalog-sync on a heading-less .rst yields ONE
    # file_index doc tagged language='rst', NO element docs, discoverable.
    # Embeddings are stubbed so no network / API key is needed.
    def test_catalog_sync_indexes_rst_as_file_index(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        _stub_embeddings(monkeypatch)
        install_test_config(monkeypatch, tmp_path, 'widgetlib')

        src = tmp_path / 'src'
        src.mkdir()
        rst = src / 'overview.rst'
        rst.write_text('Just an overview paragraph, no headings.\n', encoding='utf-8')

        library = Library(tmp_path / 'library.db')
        try:
            async def go():
                async with LibraryWriter(library) as writer:
                    return await sync_file_catalog(
                        library, writer, 'widgetlib', src, rst,
                    )

            summary = asyncio.run(go())
            assert summary.skipped is False
            assert summary.added == 0  # heading-less: no element docs

            scoped = make_scoped_library(get_config(), library, 'widgetlib')
            hits = scoped.find_documents_by_source_files(['overview.rst'])
            index_docs = [d for d in hits if d.metadata.get('kind') == 'file_index']
            assert len(index_docs) == 1
            doc = index_docs[0]
            assert doc.metadata.get('language') == 'rst'  # tagged rst, not 'unknown'
            assert doc.metadata.get('element_ids') == []  # no element docs
            assert 'overview.rst' in doc.title           # searchable by name
        finally:
            library.close()

    # ---- Slices 2.1-2.3 — docutils section extraction + nesting ------
    # ``_extract_rst`` parses the doctree and emits one ``rst_section``
    # element per section, with nested sections linked to their parent and
    # colliding title-slugs disambiguated. Wired into ``extract_elements``.
    # The page's lone top title (``Widget Toolkit``) must stay a section
    # (not be promoted to a doc title), and its three same-slug subsections
    # exercise the dedup loop.
    def test_extract_rst_emits_nested_section_elements(self, tmp_path: Path) -> None:
        rst = tmp_path / 'guide.rst'
        rst.write_text(
            'Widget Toolkit\n==============\n\nintro\n\n'
            'Set Up\n------\n\na\n\n'
            'Set-Up\n------\n\nb\n\n'
            'Set_Up\n------\n\nc\n',  # 3 distinct titles, same slug -> deduped
            encoding='utf-8',
        )
        els = extract_elements(rst, source_root=tmp_path)
        assert len(els) == 4
        assert all(e.subtype == 'rst_section' and e.language == 'rst' for e in els)
        top, subs = els[0], els[1:]
        assert top.parent_qualified_name == 'guide'                # top-level -> module
        assert all(s.parent_qualified_name == top.qualified_name for s in subs)  # nested
        assert len({s.qualified_name for s in subs}) == 3          # 3 slugs deduped
        assert [e.line_start for e in els] == sorted(e.line_start for e in els)  # order

    # No sections (empty or heading-less prose) degrades to [] — the
    # file-index fallback, explicit and never silent.
    def test_extract_rst_degrades_when_no_sections(self) -> None:
        assert catalog_extractor._extract_rst('', Path('x.rst'), Path('.')) == []
        assert catalog_extractor._extract_rst(
            'just a paragraph\n', Path('x.rst'), Path('.'),
        ) == []

    # A docutils parse failure also degrades to [] (never crashes the sync).
    def test_extract_rst_degrades_on_parse_error(self, monkeypatch) -> None:
        def boom(*a, **k):
            raise RuntimeError('docutils failed')

        monkeypatch.setattr('docutils.core.publish_doctree', boom)
        assert catalog_extractor._extract_rst(
            'Title\n=====\n', Path('x.rst'), Path('.'),
        ) == []

    # ``rst_section`` must be a valid ``Subtype`` Literal value.
    def test_rst_section_in_subtype_literal(self) -> None:
        assert 'rst_section' in get_args(Subtype)

    # ---- Slice 2.4 — sections through catalog-sync (e2e) ------------
    # End-to-end: catalog-sync on an rst WITH sections produces one element
    # doc per section (added == #sections) alongside the file_index, whose
    # element_ids reference them — no crash. (Composition check of 2.1-2.3
    # through the sync pipeline; embeddings stubbed, no network.)
    def test_catalog_sync_extracts_rst_sections(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        _stub_embeddings(monkeypatch)
        install_test_config(monkeypatch, tmp_path, 'widgetlib')

        src = tmp_path / 'src'
        src.mkdir()
        rst = src / 'guide.rst'
        rst.write_text(
            'Buttons\n=======\n\nbutton docs\n\nSliders\n=======\n\nslider docs\n',
            encoding='utf-8',
        )

        library = Library(tmp_path / 'library.db')
        try:
            async def go():
                async with LibraryWriter(library) as writer:
                    return await sync_file_catalog(
                        library, writer, 'widgetlib', src, rst,
                    )

            summary = asyncio.run(go())
            assert summary.skipped is False
            assert summary.added == 2  # one element doc per section

            scoped = make_scoped_library(get_config(), library, 'widgetlib')
            hits = scoped.find_documents_by_source_files(['guide.rst'])
            index_docs = [d for d in hits if d.metadata.get('kind') == 'file_index']
            assert len(index_docs) == 1
            assert len(index_docs[0].metadata.get('element_ids', [])) == 2
        finally:
            library.close()

    # ---- Slice 3.1 — autodoc directive capture ----------------------
    # Sphinx autodoc directives (`.. automodule:: pkg.mod`) reference code
    # symbols. docutils doesn't recognize them, so we line-scan the source
    # and attach each directive's dotted target to its enclosing section.
    # (Tier 3.2 resolves these against the SCIP index.)
    def test_extract_rst_captures_autodoc_targets(self, tmp_path: Path) -> None:
        rst = tmp_path / 'api.rst'
        rst.write_text(
            'Queue API\n=========\n\nThe queue subsystem.\n\n'
            '.. automodule:: acme.queue\n   :members:\n\n'
            '.. autoclass:: acme.queue.JobQueue\n\n'
            'Other\n=====\n\nno directives here\n',
            encoding='utf-8',
        )
        els = extract_elements(rst, source_root=tmp_path)
        by_slug = {e.qualified_name.rsplit('.', 1)[-1]: e for e in els}
        assert by_slug['queue_api'].autodoc_targets == (
            'acme.queue', 'acme.queue.JobQueue',
        )
        assert by_slug['other'].autodoc_targets == ()  # section without directives

    # ---- Slice 3.2 — resolve autodoc targets against the SCIP graph --
    # Each captured target is resolved via CrossSourceGraph: a hit links the
    # rst section to the real code symbol; a miss is recorded resolved=False
    # (the "docs reference X, but it's not in the code" signal) — never
    # silently dropped. No graph (source not SCIP-indexed) -> no links.
    def test_resolve_autodoc_links_against_scip(self, tmp_path: Path) -> None:
        sym = 'scip-python python acme 0.1 acme/queue/JobQueue#'
        doc = _ScipDoc(
            relative_path='acme/queue.py',
            occurrences=(_ScipOccurrence(symbol=sym, range=(0, 6, 0, 20), is_definition=True),),
            symbols=(_ScipSymbol(symbol=sym, kind='Class', display_name='JobQueue'),),
        )
        graph = CrossSourceGraph()
        graph.add_source('acme', index=ScipIndex(documents=(doc,)), language='python')
        graph.materialize()

        rst = tmp_path / 'api.rst'
        rst.write_text(
            'Queue\n=====\n\n'
            '.. autoclass:: acme.queue.JobQueue\n\n'
            '.. autofunction:: acme.queue.gone\n',
            encoding='utf-8',
        )
        els = extract_elements(rst, source_root=tmp_path)
        links = catalog_enrich._resolve_autodoc_links(els, graph)
        by_target = {lk.target: lk for lk in links}

        # resolved: the doc reference points at the real code symbol
        hit = by_target['acme.queue.JobQueue']
        assert hit.resolved is True
        assert hit.symbol_qualified_name == 'acme.queue.JobQueue'
        assert hit.symbol_file == 'acme/queue.py'
        # dangling: recorded explicitly, not dropped
        miss = by_target['acme.queue.gone']
        assert miss.resolved is False
        assert miss.symbol_qualified_name is None
        # source not SCIP-indexed -> no links (still documented elsewhere)
        assert catalog_enrich._resolve_autodoc_links(els, None) == ()

    # ---- Slice 3.3 — autodoc links surface on the file metadata -----
    # Wiring: _compute_scip_metadata carries the resolved autodoc links on
    # the file's ScipFileMetadata, so they flow into the doc-gen bundle.
    def test_compute_scip_metadata_carries_autodoc_links(self, tmp_path: Path) -> None:
        sym = 'scip-python python acme 0.1 acme/queue/JobQueue#'
        doc = _ScipDoc(
            relative_path='acme/queue.py',
            occurrences=(_ScipOccurrence(symbol=sym, range=(0, 6, 0, 20), is_definition=True),),
            symbols=(_ScipSymbol(symbol=sym, kind='Class', display_name='JobQueue'),),
        )
        graph = CrossSourceGraph()
        graph.add_source('acme', index=ScipIndex(documents=(doc,)), language='python')
        graph.materialize()

        rst = tmp_path / 'api.rst'
        rst.write_text(
            'Queue\n=====\n\n'
            '.. autoclass:: acme.queue.JobQueue\n\n'
            '.. autofunction:: acme.queue.gone\n',
            encoding='utf-8',
        )
        els = extract_elements(rst, source_root=tmp_path)
        meta = catalog_enrich._compute_scip_metadata(els, rst, tmp_path, graph)
        resolved = {lk.target: lk.resolved for lk in meta.autodoc_links}
        assert resolved == {'acme.queue.JobQueue': True, 'acme.queue.gone': False}

    # ---- Slice 3.4a — persisted reverse index (rst_autodoc_links) ---
    # Persist resolved rst->symbol links, then reverse-look-up: "which rst
    # sections document this code symbol?" — the basis for reverse-enrich.
    # Dangling links aren't stored; an undocumented symbol returns ().
    def test_rst_autodoc_links_reverse_index_round_trip(self) -> None:
        conn = sqlite3.connect(':memory:')
        try:
            scip.init_scip_schema(conn)
            links = (
                catalog_enrich.AutodocLink(
                    section_qualified_name='guide.queue_api',
                    target='acme.queue.JobQueue',
                    symbol_qualified_name='acme.queue.JobQueue',
                    symbol_file='acme/queue.py', resolved=True,
                ),
                catalog_enrich.AutodocLink(  # dangling -> not stored
                    section_qualified_name='guide.gone',
                    target='acme.queue.gone', resolved=False,
                ),
            )
            scip.persist_rst_autodoc_links(conn, 'acme', links)
            assert scip.rst_sections_documenting(conn, 'acme.queue.JobQueue') == ('guide.queue_api',)
            assert scip.rst_sections_documenting(conn, 'acme.queue.gone') == ()
        finally:
            conn.close()

    # ---- Slice 3.4c-1 — graph loads the reverse autodoc index -------
    # CrossSourceGraph.load_from also ingests rst_autodoc_links, so the
    # enrich path (which holds the graph, not a conn) can reverse-look-up
    # via graph.rst_sections_documenting(symbol_qn).
    def test_graph_load_from_exposes_rst_sections(self) -> None:
        conn = sqlite3.connect(':memory:')
        try:
            scip.init_scip_schema(conn)
            scip.persist_rst_autodoc_links(conn, 'acme', (
                catalog_enrich.AutodocLink(
                    section_qualified_name='guide.queue_api',
                    target='acme.queue.JobQueue',
                    symbol_qualified_name='acme.queue.JobQueue', resolved=True),
            ))
            graph = CrossSourceGraph()
            graph.load_from(conn)
            assert graph.rst_sections_documenting('acme.queue.JobQueue') == ('guide.queue_api',)
            assert graph.rst_sections_documenting('acme.queue.Unknown') == ()
        finally:
            conn.close()

    # ---- Slice 3.4c-2 — inject reverse links into code-file metadata -
    # _compute_scip_metadata, for each resolved code symbol, attaches the
    # rst sections documenting it (documented_by_rst) — the reverse-enrich
    # the code doc-gen prompt consumes. A resolved-but-undocumented symbol
    # contributes nothing (covers the inject loop's skip path).
    def test_compute_scip_metadata_carries_documented_by_rst(self) -> None:
        conn = sqlite3.connect(':memory:')
        try:
            scip.init_scip_schema(conn)
            for cid, qn in (('s1', 'acme.queue.JobQueue'), ('s2', 'acme.queue.Plain')):
                conn.execute(
                    'INSERT INTO scip_symbols (canonical_id, source_name, language, '
                    'file, line_start, line_end, kind, display_name, qualified_name, '
                    'parent_qualified_name) VALUES (?,?,?,?,?,?,?,?,?,?)',
                    (cid, 'acme', 'python', 'acme/queue.py', 1, 5, 'Class',
                     qn.rsplit('.', 1)[-1], qn, None))
            scip.persist_rst_autodoc_links(conn, 'acme', (
                catalog_enrich.AutodocLink(
                    section_qualified_name='guide.queue_api', target='acme.queue.JobQueue',
                    symbol_qualified_name='acme.queue.JobQueue', resolved=True),))
            graph = CrossSourceGraph()
            graph.load_from(conn)
            els = [_code_element('acme.queue.JobQueue'), _code_element('acme.queue.Plain')]
            meta = catalog_enrich._compute_scip_metadata(
                els, Path('acme/queue.py'), Path('.'), graph)
            docs = [(lk.symbol_qualified_name, lk.rst_section_qualified_name)
                    for lk in meta.documented_by_rst]
            assert docs == [('acme.queue.JobQueue', 'guide.queue_api')]  # Plain undocumented
        finally:
            conn.close()

    # ---- Slice 3.4c-3 — populate the reverse index at persist time --
    # persist_rst_autodoc_index re-extracts the source's rst, resolves each
    # autodoc target against the materialized graph, and writes the reverse
    # index — so it's in the DB before doc-gen's load_from. The non-rst
    # catalog file exercises the suffix filter's skip path.
    def test_persist_rst_autodoc_index_resolves_and_persists(self, tmp_path) -> None:
        (tmp_path / 'guide.rst').write_text(
            'Queue API\n=========\n\n.. autoclass:: acme.queue.JobQueue\n',
            encoding='utf-8')
        (tmp_path / 'notes.md').write_text('# Notes\n\nprose\n', encoding='utf-8')
        conn = sqlite3.connect(':memory:')
        try:
            scip.init_scip_schema(conn)
            conn.execute(
                'INSERT INTO scip_symbols (canonical_id, source_name, language, '
                'file, line_start, line_end, kind, display_name, qualified_name, '
                'parent_qualified_name) VALUES (?,?,?,?,?,?,?,?,?,?)',
                ('s1', 'acme', 'python', 'acme/queue.py', 1, 5, 'Class',
                 'JobQueue', 'acme.queue.JobQueue', None))
            graph = CrossSourceGraph()
            graph.load_from(conn)
            scip_persist.persist_rst_autodoc_index(conn, tmp_path, 'acme', graph)
            # the rst section that declared the directive now documents the symbol
            assert scip.rst_sections_documenting(conn, 'acme.queue.JobQueue')
        finally:
            conn.close()

    # ---- Slice 3.4c-4 — wire the persist step into persist_all_sources
    # End-to-end through the real persist pipeline: a source with a python
    # symbol + an rst page documenting it lands a reverse-index row. Bites
    # if the hook is ever dropped from persist_all_sources.
    def test_persist_all_sources_populates_rst_autodoc_index(self, tmp_path) -> None:
        source_root = tmp_path / 'svc'
        source_root.mkdir()
        _write_manifest(source_root, [
            {'kind': 'python', 'scip_path': 'intermediate/index-python.scip'}])
        (source_root / 'guide.rst').write_text(
            'Service Guide\n=============\n\n.. autoclass:: service.Service\n',
            encoding='utf-8')

        def _factory(scip_path, *, repo, max_staleness_days):
            return _synthetic_python_index_with_one_class(repo='svc')

        db_path = tmp_path / 'ariadne.db'
        persisted = scip_persist.persist_all_sources(
            db_path, [('svc', source_root)], index_factory=_factory)
        assert persisted == 1
        conn = sqlite3.connect(db_path)
        try:
            # the rst page now documents the resolved class symbol
            assert scip.rst_sections_documenting(conn, 'service.Service')
        finally:
            conn.close()

    # ---- Slice 3.5a — surface cross-source CALLEES in the prompt -----
    # The cross-source intelligence on ScipFileMetadata must reach the
    # doc-gen prompt, not just sit in the bundle (only `callers` did
    # before). One prompt grown field by field across 3.5a/b/c.
    def test_prompts_surface_cross_source_context(self) -> None:
        gen = DocGenerator(config=GeneratorConfig(provider='openai'))
        code_scip = catalog_enrich.ScipFileMetadata(
            callees=(catalog_enrich.ScipCallee(
                local_qualified_name='svc.queue.JobQueue.run',
                remote_qualified_name='otherlib.db.connect',
                remote_source_name='otherlib',
                remote_file='otherlib/db.py', remote_line=10),),
            documented_by_rst=(catalog_enrich.ReverseAutodocLink(
                symbol_qualified_name='svc.queue.JobQueue',
                rst_section_qualified_name='guide.queue_api'),),
        )
        code_bundle = catalog_enrich.EnrichedFileBundle(
            path=Path('svc/queue.py'), language='python',
            module_name='svc.queue', scip=code_scip)
        arch = gen._format_prompt_from_bundle(ARCHITECTURE_PROMPT, code_bundle, 'CODE')
        # 3.5a — outbound cross-source calls surface in the architecture prompt
        assert 'otherlib.db.connect' in arch
        # no scip → the section degrades to a placeholder (False arm), no crash
        bare = catalog_enrich.EnrichedFileBundle(
            path=Path('svc/bare.py'), language='python', module_name='svc.bare')
        assert '(None detected)' in gen._format_prompt_from_bundle(
            ARCHITECTURE_PROMPT, bare, 'CODE')
        # legacy/group render sites omit cross_source_calls — the new slot
        # must default, not KeyError (regression guard for the shared template)
        assert '(None detected)' in render_user_template(
            ARCHITECTURE_PROMPT, language='python', component_info='c',
            source_code='s', dependencies='d', dependents='dep')

        # 3.5b — reverse: the rst section documenting this code surfaces in
        # the explanation prompt (the human rationale to fold in)
        code_expl = gen._format_prompt_from_bundle(EXPLANATION_PROMPT, code_bundle, 'CODE')
        assert 'guide.queue_api' in code_expl
        assert gen._format_prompt_from_bundle(EXPLANATION_PROMPT, bare, 'CODE')  # scip None: no crash
        # 3.5c — forward: an rst page surfaces the code symbols it documents;
        # unresolved targets are omitted (not fact-checkable against code)
        rst_scip = catalog_enrich.ScipFileMetadata(
            autodoc_links=(
                catalog_enrich.AutodocLink(
                    section_qualified_name='guide.queue_api', target='svc.queue.JobQueue',
                    symbol_qualified_name='svc.queue.JobQueue', resolved=True),
                catalog_enrich.AutodocLink(
                    section_qualified_name='guide.misc', target='svc.queue.Gone',
                    symbol_qualified_name=None, resolved=False),
            ),
        )
        rst_bundle = catalog_enrich.EnrichedFileBundle(
            path=Path('docs/guide.rst'), language='rst',
            module_name='docs.guide', scip=rst_scip)
        rst_expl = gen._format_prompt_from_bundle(EXPLANATION_PROMPT, rst_bundle, 'RST')
        assert 'svc.queue.JobQueue' in rst_expl       # resolved target surfaces
        assert 'svc.queue.Gone' not in rst_expl       # unresolved omitted
        # scip present but no rst links → empty block, no bare header (`not lines` arm)
        calls_only = catalog_enrich.ScipFileMetadata(callees=code_scip.callees)
        assert format_rst_crossref(calls_only) == ''


def _code_element(qualified_name: str) -> ElementInfo:
    """A minimal code ElementInfo for reverse-autodoc tests."""
    return ElementInfo(
        language='python', subtype='class', file='acme/queue.py',
        qualified_name=qualified_name, signature='', line_start=1,
        line_end=1, col_start=0, col_end=0,
    )


def _stub_embeddings(monkeypatch) -> None:
    """Replace the embedding service so the writer needs no API key/network."""
    async def fake_embed(self, text):
        return np.zeros(3072, dtype=np.float32)

    async def fake_embed_batch(self, texts):
        return [np.zeros(3072, dtype=np.float32) for _ in texts]

    async def fake_get_client(self):
        return None

    async def fake_close(self):
        return None

    monkeypatch.setattr('embedding.EmbeddingService.embed', fake_embed)
    monkeypatch.setattr('embedding.EmbeddingService.embed_batch', fake_embed_batch)
    monkeypatch.setattr('embedding.EmbeddingService._get_client', fake_get_client)
    monkeypatch.setattr('embedding.EmbeddingService.close', fake_close)
