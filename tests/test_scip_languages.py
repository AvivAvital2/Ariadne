"""Contract for the SCIP language registry (Phase 2j.b).

`docgen/scip_languages.py` is the single source of truth for SCIP
language metadata: which extensions belong to which language, which
marker files signal a "package," whether the indexer can handle
standalone files, and the indexer-kind label that dispatches to the
right adapter.

Adding a new language (e.g., Go, Ruby) should be a single
:class:`LanguageDef` entry — no edits to ``discover()``, no edits to
the indexer adapter registry. Tests below pin that contract.

These tests are RED until ``docgen/scip_languages.py`` exists.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Registry contents
# ---------------------------------------------------------------------------


class TestRegistryContents:
    def test_python_in_registry(self) -> None:
        """Python: ``.py`` source extension, ``__init__.py`` marker,
        indexer_kind='python', standalone-capable (pyright handles
        single-file modules)."""
        from docgen.scip_languages import LANGUAGES

        python = next((l for l in LANGUAGES if l.name == 'python'), None)
        assert python is not None, 'python missing from LANGUAGES'
        assert '.py' in python.source_extensions
        assert '__init__.py' in python.marker_files
        assert python.indexer_kind == 'python'
        assert python.can_index_standalone is True

    def test_typescript_in_registry(self) -> None:
        """TypeScript: covers .ts/.tsx + .js/.jsx/.mjs/.cjs (scip-typescript
        handles modern JS too), marker is ``package.json``, standalone
        via ``--infer-tsconfig``."""
        from docgen.scip_languages import LANGUAGES

        ts = next((l for l in LANGUAGES if l.name == 'typescript'), None)
        assert ts is not None, 'typescript missing from LANGUAGES'
        for ext in ('.ts', '.tsx', '.js', '.jsx', '.mjs'):
            assert ext in ts.source_extensions, f'{ext} missing'
        assert 'package.json' in ts.marker_files
        assert ts.indexer_kind == 'typescript'
        assert ts.can_index_standalone is True

    def test_jvm_in_registry(self) -> None:
        """JVM: covers .scala/.java/.kt, multiple build-tool markers,
        indexer_kind='java' (scip-java handles all JVM languages),
        and crucially NOT standalone-capable — scip-java needs a build
        root (sbt/Maven/Gradle) to compile."""
        from docgen.scip_languages import LANGUAGES

        jvm = next((l for l in LANGUAGES if l.name == 'jvm'), None)
        assert jvm is not None, 'jvm missing from LANGUAGES'
        assert '.scala' in jvm.source_extensions
        assert '.java' in jvm.source_extensions
        for marker in ('build.sbt', 'pom.xml', 'build.gradle'):
            assert marker in jvm.marker_files, f'{marker} missing'
        assert jvm.indexer_kind == 'java'
        assert jvm.can_index_standalone is False, (
            'JVM indexer needs build tool — cannot handle orphan files'
        )


# ---------------------------------------------------------------------------
# Registry invariants (caught at module load — pin them as tests so a
# regression in one entry doesn't slip past)
# ---------------------------------------------------------------------------


class TestRegistryInvariants:
    def test_indexer_kinds_are_unique(self) -> None:
        """No two languages share an indexer_kind. Otherwise the adapter
        registry can't map kind → adapter unambiguously."""
        from docgen.scip_languages import LANGUAGES

        kinds = [l.indexer_kind for l in LANGUAGES]
        assert len(kinds) == len(set(kinds)), (
            f'duplicate indexer_kind values: {kinds}'
        )

    def test_language_names_are_unique(self) -> None:
        """Language names are the public registry key — must be unique."""
        from docgen.scip_languages import LANGUAGES

        names = [l.name for l in LANGUAGES]
        assert len(names) == len(set(names))

    def test_extensions_are_lowercase_dot_prefixed(self) -> None:
        """Invariant for ``_EXT_TO_LANG`` lookups: every extension is
        lowercase and starts with a dot. Mismatches here silently
        break file matching."""
        from docgen.scip_languages import LANGUAGES

        for lang in LANGUAGES:
            for ext in lang.source_extensions:
                assert ext.startswith('.'), (
                    f'{lang.name}: extension {ext!r} missing leading dot'
                )
                assert ext == ext.lower(), (
                    f'{lang.name}: extension {ext!r} not lowercase'
                )

    def test_marker_files_are_basenames_not_paths(self) -> None:
        """Markers are file *names* checked per directory entry —
        no slashes allowed, otherwise path comparison breaks."""
        from docgen.scip_languages import LANGUAGES

        for lang in LANGUAGES:
            for marker in lang.marker_files:
                assert '/' not in marker
                assert '\\' not in marker

    def test_source_extensions_dont_overlap_across_languages(self) -> None:
        """A source extension belongs to exactly one language. If two
        languages claim ``.js``, ``_EXT_TO_LANG`` is ambiguous."""
        from docgen.scip_languages import LANGUAGES

        seen: dict[str, str] = {}
        for lang in LANGUAGES:
            for ext in lang.source_extensions:
                assert ext not in seen, (
                    f'extension {ext!r} claimed by both '
                    f'{seen[ext]} and {lang.name}'
                )
                seen[ext] = lang.name


# ---------------------------------------------------------------------------
# Derived lookup tables — exposed so callers don't rebuild them
# ---------------------------------------------------------------------------


class TestDerivedLookups:
    def test_extension_lookup_maps_to_language(self) -> None:
        """``_EXT_TO_LANG`` (or equivalent helper) lets a caller resolve
        ``.py`` → Python, ``.ts`` → TypeScript, ``.scala`` → JVM in O(1).
        Used by ``discover()`` during the single tree walk."""
        from docgen.scip_languages import _EXT_TO_LANG

        assert _EXT_TO_LANG['.py'].name == 'python'
        assert _EXT_TO_LANG['.tsx'].name == 'typescript'
        assert _EXT_TO_LANG['.scala'].name == 'jvm'
        assert _EXT_TO_LANG.get('.unknown') is None

    def test_marker_lookup_maps_to_language(self) -> None:
        """``_MARKER_TO_LANG`` lets a caller resolve marker file names
        to their language. Used during the same single walk."""
        from docgen.scip_languages import _MARKER_TO_LANG

        assert _MARKER_TO_LANG['__init__.py'].name == 'python'
        assert _MARKER_TO_LANG['package.json'].name == 'typescript'
        assert _MARKER_TO_LANG['build.sbt'].name == 'jvm'
        assert _MARKER_TO_LANG['pom.xml'].name == 'jvm'
        assert _MARKER_TO_LANG.get('Makefile') is None


# ---------------------------------------------------------------------------
# LanguageDef shape
# ---------------------------------------------------------------------------


class TestLanguageDefShape:
    def test_language_def_is_frozen(self) -> None:
        """LanguageDef is an attrs @frozen value object — immutable so
        callers can pass it around without defensive copies and tests
        can compare by equality."""
        from docgen.scip_languages import LANGUAGES, LanguageDef

        assert all(isinstance(l, LanguageDef) for l in LANGUAGES)
        # @frozen raises on attribute assignment
        try:
            LANGUAGES[0].name = 'mutated'  # type: ignore[misc]
        except Exception:
            return
        raise AssertionError(
            'LanguageDef should be frozen (attrs @frozen)',
        )

    def test_extension_and_marker_sets_are_frozen(self) -> None:
        """source_extensions and marker_files are frozensets — pinning
        immutability all the way down so callers can use them as dict
        keys or set members without subtle aliasing."""
        from docgen.scip_languages import LANGUAGES

        for lang in LANGUAGES:
            assert isinstance(lang.source_extensions, frozenset), (
                f'{lang.name}.source_extensions not a frozenset'
            )
            assert isinstance(lang.marker_files, frozenset), (
                f'{lang.name}.marker_files not a frozenset'
            )
