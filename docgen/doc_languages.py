"""Doc-generation language detection — file extension / name → language label.

The single source for "what language is this file, for doc-generation purposes",
shared by catalog extraction (:mod:`docgen.catalog_extractor`) and cost
estimation (:mod:`docgen.pricing`) so the two can't disagree on a file's
language (they had drifted — pricing was missing ``.vue``/``.conf``/``.css``).

Deliberately DISTINCT from :mod:`docgen.scip_languages`, which is the registry
of SCIP-indexable *code* languages only. This map is broader — it also covers
config/prose that the doc generator handles but SCIP does not (hocon, css, rst,
html, json, yaml, markdown, dockerfile). Kept dependency-light (stdlib only) so
either consumer can import it without pulling the heavy extractor stack.
"""
from __future__ import annotations

from pathlib import Path

# Extension → doc-language label. ``.vue`` maps to javascript: its
# ``.vue.script.{js,ts}`` companion is what the SCIP path indexes, and the doc
# generator treats the SFC as JavaScript. Dockerfile is matched by name below
# (it has no extension), plus the ``.dockerfile`` extension here.
_EXT_TO_DOC_LANGUAGE: dict[str, str] = {
    '.py': 'python',
    '.html': 'html', '.htm': 'html',
    '.js': 'javascript', '.jsx': 'javascript', '.ts': 'javascript',
    '.tsx': 'javascript', '.mjs': 'javascript', '.vue': 'javascript',
    '.json': 'json',
    '.yaml': 'yaml', '.yml': 'yaml',
    '.md': 'markdown', '.markdown': 'markdown',
    '.scala': 'scala', '.sbt': 'scala',
    '.java': 'java',
    '.go': 'go',
    '.conf': 'hocon',
    '.css': 'css',
    '.rst': 'rst',
    '.dockerfile': 'dockerfile',
}


def detect_doc_language(path) -> str | None:
    """The doc-generation language for ``path`` (by filename then extension),
    or ``None`` when the file type isn't one the doc generator handles."""
    p = Path(path)
    name = p.name
    if name == 'Dockerfile' or name.startswith('Dockerfile.'):
        return 'dockerfile'
    return _EXT_TO_DOC_LANGUAGE.get(p.suffix.lower())
