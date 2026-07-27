"""Cost estimation and catalog extraction detect the same doc language for a
file — one shared table (docgen.doc_languages), not a hand-copied mirror that
drifts. Pricing gains the .vue/.conf/.css cases catalog already had.
"""
from __future__ import annotations

from pathlib import Path

import docgen.catalog_extractor as ce
import docgen.pricing as pricing
from docgen.doc_languages import detect_doc_language

_CASES = [
    'a.py', 'a.ts', 'a.tsx', 'a.jsx', 'a.mjs', 'a.js', 'a.vue',
    'a.scala', 'a.sbt', 'a.java', 'a.go',
    'a.conf', 'a.css', 'a.rst', 'a.md', 'a.markdown',
    'a.json', 'a.yaml', 'a.yml', 'a.html',
    'Dockerfile', 'Dockerfile.prod', 'x.unknown',
]


def test_shared_table_values():
    assert detect_doc_language(Path('a.conf')) == 'hocon'
    assert detect_doc_language(Path('a.css')) == 'css'
    assert detect_doc_language(Path('a.vue')) == 'javascript'
    assert detect_doc_language(Path('Dockerfile')) == 'dockerfile'
    assert detect_doc_language(Path('a.xyz')) is None


def test_pricing_and_catalog_agree_on_every_case():
    for name in _CASES:
        p = Path(name)
        assert pricing._detect_language(p) == ce._detect_language(p), name


def test_pricing_gained_the_drifted_cases():
    # The exact drift that silently mispriced config/SFC files.
    assert pricing._detect_language(Path('x.conf')) == 'hocon'
    assert pricing._detect_language(Path('x.css')) == 'css'
    assert pricing._detect_language(Path('x.vue')) == 'javascript'
