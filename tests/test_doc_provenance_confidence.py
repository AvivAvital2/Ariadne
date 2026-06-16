"""Tier 4 — provenance confidence for human-authored docs (designs/rst-support.md).

Human docs (rst/markdown) are maintained by people, not derived from the code
we rely on, so they are treated with a grain of salt: tagged by provenance,
ranked below code-derived docs, losing conflicts to code, and surfaced with a
"no code evidence found to back this" badge only when they are the sole
evidence.

Grows one slice at a time: 4.1 config + provenance tag, 4.2 ranking weight,
4.3 conflict + badge, 4.4 stale finding.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from ariadne_mcp import service_analysis
from cli import callers
from config import (
    CODE_PROVENANCE,
    HUMAN_DOC_PROVENANCE,
    Config,
    doc_provenance,
)
from docgen import catalog_enrich, generator, orchestrator, scip_persist
from docgen.generator import GeneratedDoc
from docgen.scip_cross_source import CrossSourceGraph
from library import Library, scip
from schema import Document
from writer import LibraryWriter


class TestProvenanceConfidence:
    # ---- Slice 4.1 — config knob + provenance determination ---------
    def test_config_parses_low_confidence_doc_languages(self, tmp_path: Path) -> None:
        p = tmp_path / 'ariadne.yaml'
        p.write_text(
            'sources:\n'
            '  default_langs:\n'
            '    path: /tmp/a\n'
            '  custom_langs:\n'
            '    path: /tmp/b\n'
            '    low_confidence_doc_languages: [rst, html]\n'
            '  opted_out:\n'
            '    path: /tmp/c\n'
            '    low_confidence_doc_languages: []\n',
            encoding='utf-8')
        cfg = Config(config_path=p)
        # absent → default human-prose languages
        assert cfg.get_source_config('default_langs').low_confidence_doc_languages == ('rst', 'markdown')
        # explicit list overrides the default
        assert cfg.get_source_config('custom_langs').low_confidence_doc_languages == ('rst', 'html')
        # empty list → opt out (nothing treated as low-confidence)
        assert cfg.get_source_config('opted_out').low_confidence_doc_languages == ()

    def test_doc_provenance_from_source_language(self) -> None:
        langs = ('rst', 'markdown')
        assert doc_provenance('rst', langs) == HUMAN_DOC_PROVENANCE
        assert doc_provenance('markdown', langs) == HUMAN_DOC_PROVENANCE
        assert doc_provenance('python', langs) == CODE_PROVENANCE
        # opted-out source (empty list) → even rst is code-derived
        assert doc_provenance('rst', ()) == CODE_PROVENANCE

    # ---- Slice 4.2 — human-doc provenance sinks below code in ranking
    # The human doc is the CLOSER raw match, so without the provenance weight
    # it would rank first; the weight must reorder it below the code doc.
    def test_search_downranks_human_doc_provenance(self, tmp_path: Path) -> None:
        lib = Library(tmp_path / 'rank.db')
        try:
            e_query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            e_human = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # raw sim 1.0 (closer)
            e_code = np.array([0.9, 0.0, 0.0], dtype=np.float32)   # raw sim 0.9
            lib.add_document('explanation', 'Human doc', 'h', embedding=e_human,
                             metadata={'provenance': HUMAN_DOC_PROVENANCE})
            lib.add_document('explanation', 'Code doc', 'c', embedding=e_code,
                             metadata={'provenance': CODE_PROVENANCE})
            results = lib.search(e_query, k=2)
            titles = [r.document.title for r in results]
            # despite being the closer raw match, the human doc sinks below code
            assert titles == ['Code doc', 'Human doc']
            assert results[0].score > results[1].score
        finally:
            lib.close()

    # ---- Slice 4.2b — provenance tagged at storage from source language --
    # A doc's provenance is the language of the file(s) it documents; a doc
    # drawing on any human-authored source counts as human-doc.
    def test_doc_provenance_for_files(self) -> None:
        langs = ('rst', 'markdown')
        f = orchestrator._doc_provenance_for_files
        assert f(('docs/guide.rst',), langs) == HUMAN_DOC_PROVENANCE
        assert f(('pkg/mod.py',), langs) == CODE_PROVENANCE
        # any human-authored source file → human-doc
        assert f(('pkg/mod.py', 'docs/g.rst'), langs) == HUMAN_DOC_PROVENANCE
        # no source files → code-derived (nothing human to flag)
        assert f((), langs) == CODE_PROVENANCE

    # ---- provenance prefers the bundle's own language over re-detection -
    # The generator already stamps metadata['language']; use it directly.
    # Group/topic docs carry no single language → fall back to file detection.
    def test_provenance_prefers_bundle_language(self) -> None:
        langs = ('rst', 'markdown')

        def gd(metadata, source_files=()):
            return GeneratedDoc(title='t', content='c', doc_type='explanation',
                                source_files=source_files, metadata=metadata)

        prov = orchestrator._provenance_for_doc
        # bundle path: stamped language is authoritative (no file re-detection)
        assert prov(gd({'language': 'rst'}), langs) == HUMAN_DOC_PROVENANCE
        assert prov(gd({'language': 'python'}), langs) == CODE_PROVENANCE
        # group/topic docs (no language) fall back to the documented files
        assert prov(gd({}, ('docs/g.rst',)), langs) == HUMAN_DOC_PROVENANCE
        assert prov(gd({}, ('pkg/m.py',)), langs) == CODE_PROVENANCE

    # ---- uniform metadata-forwarding: typed writers merge forwarded metadata --
    # add_diagram/add_qa hardcoded their type-specific metadata, dropping any
    # caller metadata (e.g. provenance). They must now merge it in.
    async def test_typed_writers_forward_metadata(self, tmp_path: Path, monkeypatch) -> None:
        lib = Library(tmp_path / 'w.db')
        try:
            w = LibraryWriter(lib)
            captured: list[dict] = []

            async def fake_add_document(**kwargs):
                captured.append(dict(kwargs.get('metadata') or {}))
                return None

            monkeypatch.setattr(w, 'add_document', fake_add_document)
            await w.add_diagram(title='T', description='d', dot_code='digraph{}',
                                metadata={'provenance': HUMAN_DOC_PROVENANCE})
            await w.add_qa(question='q?', answer='a',
                           metadata={'provenance': HUMAN_DOC_PROVENANCE})
            # type-specific keys preserved AND forwarded provenance merged in
            assert captured[0].get('dot_code') == 'digraph{}'
            assert captured[0].get('provenance') == HUMAN_DOC_PROVENANCE
            assert captured[1].get('question') == 'q?'
            assert captured[1].get('provenance') == HUMAN_DOC_PROVENANCE
        finally:
            lib.close()

    # ---- Tier 4.4 — dangling autodoc detection (stale-doc signal) ------
    # An rst autodoc target that no longer resolves to a code symbol means the
    # docs reference code that's gone/renamed. ignore_staleness suppresses it.
    def test_dangling_autodoc_collects_unresolved(self, tmp_path: Path) -> None:
        (tmp_path / 'guide.rst').write_text(
            'Queue\n=====\n\n.. autoclass:: acme.queue.JobQueue\n'
            '.. autoclass:: acme.queue.Gone\n', encoding='utf-8')
        (tmp_path / 'notes.md').write_text('# Notes\n', encoding='utf-8')  # non-rst: skipped
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
            targets = [t for _, t in scip_persist.dangling_autodoc(tmp_path, graph)]
            assert 'acme.queue.Gone' in targets          # unresolved → flagged
            assert 'acme.queue.JobQueue' not in targets  # resolved → not flagged
            # ignore_staleness suppresses it (don't nag opted-out sources)
            assert scip_persist.dangling_autodoc(tmp_path, graph, ignore_staleness=True) == []
        finally:
            conn.close()

    # ---- Tier 4.4b — improve report formatter (pure: dangling list -> text)
    def test_format_stale_autodoc_report(self) -> None:
        assert callers.format_stale_autodoc_report([]) is None  # nothing → no section
        report = callers.format_stale_autodoc_report(
            [('guide.queue_api', 'acme.queue.Gone')])
        assert report is not None
        assert 'acme.queue.Gone' in report and 'guide.queue_api' in report

    # ---- Tier 4.4c — stale rst (dangling autodoc) sinks below fresh docs --
    # The stale doc is the CLOSER raw match, so without the stale weight it
    # would rank first; referencing missing code must sink it below the fresh.
    def test_search_downranks_stale_autodoc(self, tmp_path: Path) -> None:
        lib = Library(tmp_path / 's.db')
        try:
            e_query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            lib.add_document(
                'explanation', 'Stale rst', 's',
                embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),  # closer
                metadata={'provenance': HUMAN_DOC_PROVENANCE, 'stale_autodoc': True})
            lib.add_document(
                'explanation', 'Fresh rst', 'f',
                embedding=np.array([0.9, 0.0, 0.0], dtype=np.float32),
                metadata={'provenance': HUMAN_DOC_PROVENANCE})
            titles = [r.document.title for r in lib.search(e_query, k=2)]
            assert titles == ['Fresh rst', 'Stale rst']  # broken-ref doc sinks below fresh
        finally:
            lib.close()

    # ---- Tier 4.4c — tag stale_autodoc from the bundle (so the weight fires)
    def test_dangling_autodoc_meta(self) -> None:
        def bundle(autodoc_links):
            return catalog_enrich.EnrichedFileBundle(
                path=Path('g.rst'), language='rst', module_name='g',
                scip=catalog_enrich.ScipFileMetadata(autodoc_links=autodoc_links))

        dangling = (catalog_enrich.AutodocLink(
            section_qualified_name='g.s', target='a.Gone',
            symbol_qualified_name=None, resolved=False),)
        resolved = (catalog_enrich.AutodocLink(
            section_qualified_name='g.s', target='a.X',
            symbol_qualified_name='a.X', resolved=True),)
        assert generator._dangling_autodoc_meta(bundle(dangling)) == {'stale_autodoc': True}
        assert generator._dangling_autodoc_meta(bundle(resolved)) == {}
        # code docs (no scip / no autodoc) → no tag
        nobundle = catalog_enrich.EnrichedFileBundle(
            path=Path('m.py'), language='python', module_name='m')
        assert generator._dangling_autodoc_meta(nobundle) == {}

    # ---- Tier 4.3a — conflict resolution: code-derived beats human-doc ----
    # Equal timestamps so the recency tiebreaker can't decide — the provenance
    # rule must pick code in BOTH argument orders.
    def test_conflict_resolution_code_wins_over_human(self, tmp_path: Path) -> None:
        lib = Library(tmp_path / 'c.db')
        try:
            ts = '2026-01-01T00:00:00'
            human = Document(content_type='explanation', title='X', content='c',
                             metadata={'provenance': HUMAN_DOC_PROVENANCE}, updated_at=ts)
            code = Document(content_type='explanation', title='X', content='c',
                            metadata={'provenance': CODE_PROVENANCE}, updated_at=ts)
            assert lib._resolve_conflict_pair(human, code, None, None) is code
            assert lib._resolve_conflict_pair(code, human, None, None) is code
        finally:
            lib.close()

    # ---- Tier 4.3b — 📄 badge ONLY when the answer is sole-source human-doc
    def test_doc_only_badge(self) -> None:
        def doc(prov):
            return Document(content_type='explanation', title='t', content='c',
                            metadata=({'provenance': prov} if prov else {}))

        badge = service_analysis._doc_only_badge
        out = badge([doc(HUMAN_DOC_PROVENANCE), doc(HUMAN_DOC_PROVENANCE)])
        assert '📄' in out and 'no code evidence' in out.lower()
        # any code-derived corroboration → no badge
        assert badge([doc(HUMAN_DOC_PROVENANCE), doc(CODE_PROVENANCE)]) == ''
        # untagged docs count as non-human (conservative) → no badge
        assert badge([doc(HUMAN_DOC_PROVENANCE), doc(None)]) == ''
        assert badge([]) == ''
