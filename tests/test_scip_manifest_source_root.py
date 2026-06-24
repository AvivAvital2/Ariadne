"""Regression: multi-package SCIP ``source_root`` resolution. A per-package
``.scip`` lives under ``.ariadne/intermediate``, but its
document paths are relative to the indexer's *cwd* (the package root, e.g.
``be``). ``load_source_from_manifest`` must set the loaded index's ``source_root``
to ``<repo>/<cwd>`` so source-file reads (ORM strategies, etc.) resolve — NOT the
``.scip`` file's own directory (where source never lives). Before the fix,
source-reading strategies got 'did not parse' for every file on a multi-package
repo. Synthetic fixtures only.
"""
from __future__ import annotations

import json
from pathlib import Path

from docgen.scip_cross_source import CrossSourceGraph, load_source_from_manifest
from docgen.scip_extractor import ScipIndex, _ScipDoc


def test_manifest_sets_source_root_to_package_cwd(tmp_path):
    # multi-package layout: the python package lives in <repo>/be; its .scip is
    # under <repo>/.ariadne/intermediate; document paths are package-relative.
    (tmp_path / 'be').mkdir()
    (tmp_path / 'be' / 'models.py').write_text('class User: pass\n')
    (tmp_path / '.ariadne' / 'intermediate').mkdir(parents=True)
    (tmp_path / '.ariadne' / 'manifest.json').write_text(json.dumps({
        'indexers': [{'kind': 'python', 'cwd': 'be',
                      'scip_path': 'intermediate/index-be.scip'}]}))

    def fake_loader(scip_path, repo=None, max_staleness_days=None):
        # ScipIndex.load returns source_root = the .scip's OWN dir (the bug);
        # the document path is package-relative ('models.py', no 'be/').
        return ScipIndex(
            documents=(_ScipDoc(relative_path='models.py', occurrences=(), symbols=()),),
            source_root=scip_path.parent)

    graph = CrossSourceGraph()
    load_source_from_manifest(graph, 'src', tmp_path, index_factory=fake_loader)

    idx = graph._sources['src'][0].index
    # corrected to the package root, so <root>/<doc path> is the REAL file —
    # not <repo>/.ariadne/intermediate/models.py (the bug's value).
    assert idx.source_root == tmp_path / 'be'
    assert (idx.source_root / idx.documents[0].relative_path).read_text() == 'class User: pass\n'
