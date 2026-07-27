"""SCIP language registry — single source of truth for language metadata.

Adding a new language (e.g., Go, Ruby) to SCIP discovery and indexing
should be a single :class:`LanguageDef` entry in :data:`LANGUAGES`. No
edits to ``discover()``, no per-language regex blocks, no hand-rolled
extension lists scattered across the codebase.

Each language carries:

- ``name`` — registry key (e.g., ``'python'``, ``'typescript'``, ``'jvm'``)
- ``source_extensions`` — file extensions the indexer can process
- ``marker_files`` — filenames that signal a "package" rooted at that dir
- ``indexer_kind`` — the IndexerKind label for adapter dispatch in
  ``cli_core._default_indexer_registry``
- ``can_index_standalone`` — whether the indexer can handle standalone
  source files without a build/package marker. Python (pyright) and
  TypeScript (``--infer-tsconfig``) say yes. JVM (scip-java) says no —
  it requires sbt/Maven/Gradle to compile.

The orthogonality matters:

- ``name`` is internal, language-as-a-concept.
- ``indexer_kind`` is the dispatch label — for JVM this is ``'java'``
  because scip-java covers Scala + Java + Kotlin under that one
  indexer's umbrella.

The two derived lookup tables (``_EXT_TO_LANG`` and ``_MARKER_TO_LANG``)
are built once at module import. Discovery uses them in O(1) per file
during a single tree walk.
"""
from __future__ import annotations

from attrs import frozen


@frozen
class LanguageDef:
    """Compile-time facts about a SCIP-indexable language."""
    name: str
    source_extensions: frozenset[str]
    marker_files: frozenset[str]
    indexer_kind: str
    can_index_standalone: bool
    # Human names for this language that aren't the internal ``name``: the
    # umbrella-indexer members (jvm ← java/scala/kotlin) and common synonyms
    # (typescript ← javascript, go ← golang). One source for name-based
    # lookups (e.g. spool eligibility) so they can't drift from the registry.
    aliases: frozenset[str] = frozenset()


LANGUAGES: tuple[LanguageDef, ...] = (
    LanguageDef(
        name='python',
        source_extensions=frozenset({'.py'}),
        marker_files=frozenset({'__init__.py'}),
        indexer_kind='python',
        # pyright handles single-file modules via include patterns —
        # no package boundary required.
        can_index_standalone=True,
    ),
    LanguageDef(
        name='typescript',
        # scip-typescript handles modern JS extensions too; .cjs / .mjs
        # are the CommonJS / ES module variants. .vue is routed here so a
        # Vue-only directory registers as a TS scope; the Vue extractor
        # turns each SFC into a *.vue.script.{js,ts} companion that
        # scip-typescript then indexes.
        source_extensions=frozenset({
            '.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs', '.vue',
        }),
        marker_files=frozenset({'package.json'}),
        indexer_kind='typescript',
        # --infer-tsconfig lets scip-typescript work without a real
        # tsconfig.json — handles standalone files.
        can_index_standalone=True,
        aliases=frozenset({'javascript'}),
    ),
    LanguageDef(
        name='jvm',
        # scip-java is one indexer covering all JVM source-file types.
        source_extensions=frozenset({'.scala', '.java', '.kt'}),
        marker_files=frozenset({
            'build.sbt', 'pom.xml',
            'build.gradle', 'build.gradle.kts',
        }),
        indexer_kind='java',
        # scip-java needs the build tool to compile sources first —
        # there's no equivalent of pyright's standalone mode. Orphan
        # .scala / .java files cannot be indexed by Ariadne via this
        # path. Layer C may still target them as cross-language
        # endpoint files.
        can_index_standalone=False,
        aliases=frozenset({'java', 'scala', 'kotlin'}),
    ),
    LanguageDef(
        name='go',
        source_extensions=frozenset({'.go'}),
        marker_files=frozenset({'go.mod'}),
        indexer_kind='go',
        # scip-go type-checks a Go module via go/packages, so it needs a
        # go.mod-rooted module — no standalone single-file mode. Each go.mod
        # is its own scope (like a JVM build marker), but unlike scip-java
        # there's no build-tool lifecycle to orchestrate: the Go toolchain
        # compiles fast, so indexing is a single quick pass per module.
        can_index_standalone=False,
        aliases=frozenset({'golang'}),
    ),
)


# Derived lookups — computed once at module load. Use these instead of
# rebuilding the dict in every caller.
_EXT_TO_LANG: dict[str, LanguageDef] = {
    ext: lang for lang in LANGUAGES for ext in lang.source_extensions
}

_MARKER_TO_LANG: dict[str, LanguageDef] = {
    marker: lang
    for lang in LANGUAGES
    for marker in lang.marker_files
}


# --- Tree-sitter grammar extension sets (SCIP extractor file-filters) --------
# The SCIP-driven extractors parse source with a tree-sitter grammar to read
# call structure / string literals. Which files each grammar handles is defined
# ONCE here so no extractor hand-rolls (and drifts) its own list — and so a file
# is never handed to the wrong language's grammar.

_TYPESCRIPT = next(lang for lang in LANGUAGES if lang.name == 'typescript')

# The 'javascript' grammar parses JS and TS alike (scip-typescript indexes both;
# the grammar parses TS leniently — a deliberate, working reuse). Derived from
# the typescript indexer entry, minus .vue: a raw .vue isn't valid JS, so the
# Vue path emits a .vue.script.{js,ts} companion that this grammar parses.
JS_GRAMMAR_EXTS: frozenset[str] = _TYPESCRIPT.source_extensions - frozenset({'.vue'})

# The 'scala' grammar parses .scala plus .sbt build definitions (also Scala
# syntax). NOT derived from the jvm indexer entry: that bucket also covers
# .java/.kt (one scip-java indexer), and feeding those to tree-sitter-scala
# would conflate languages. .sbt is not an indexer source extension. Deriving
# this would require splitting jvm into scala/java/kotlin — a larger change.
SCALA_GRAMMAR_EXTS: frozenset[str] = frozenset({'.scala', '.sbt'})

PY_GRAMMAR_EXTS: frozenset[str] = next(
    lang for lang in LANGUAGES if lang.name == 'python').source_extensions
GO_GRAMMAR_EXTS: frozenset[str] = next(
    lang for lang in LANGUAGES if lang.name == 'go').source_extensions


__all__ = [
    'GO_GRAMMAR_EXTS',
    'JS_GRAMMAR_EXTS',
    'LANGUAGES',
    'LanguageDef',
    'PY_GRAMMAR_EXTS',
    'SCALA_GRAMMAR_EXTS',
    '_EXT_TO_LANG',
    '_MARKER_TO_LANG',
]
