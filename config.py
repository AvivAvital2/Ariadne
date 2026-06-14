"""Configuration file handling for Ariadne.

This module provides config file loading from ariadne.yaml,
supporting both local and global configuration.
"""
from __future__ import annotations

__all__ = [
    'Config',
    'ConfigError',
    'DEFAULT_EXCLUDE_POLICY',
    'DEFAULT_EXCLUDE_FILE_PATTERNS',
    'SourceConfig',
    'get_config',
    'reload_config',
]

import difflib
import os
from pathlib import Path
from typing import Any

import yaml
from attrs import field, frozen


# Recognized keys for a dict-form source entry in ariadne.yaml. Used by
# the loader's typo-detection to fail loud on unknown keys with a
# difflib-based suggestion (e.g., ``purh:`` → "did you mean 'path'?").
# Anything not in this set raises ConfigError at config load.
_SOURCE_KNOWN_KEYS: frozenset[str] = frozenset({
    'path',
    'depends_on',
    'parent',
    'branches',
    'ref',
    'exclude',
    'exclude_dirs',
    'exempt_dirs',
    'env_hints',
    'swagger_paths',
    # SCIP-related keys consumed by get_source_scip_config
    'scip',
    'index_kinds','ignore_staleness'
})


class ConfigError(ValueError):
    """Raised when ariadne.yaml fails validation at load time.

    Currently used for contradictions between per-source ``exclude_dirs``
    and ``exempt_dirs`` (a dir cannot both be skipped and force-walked).
    """


# Canonical default set of directory names pruned at every depth during
# discovery (find_python_files / find_catalog_files / iter_catalog_files).
#
# Top-level ``exclude_policy:`` in ariadne.yaml REPLACES this list (set
# it to ``[]`` for a full walk). Per-source ``exclude_dirs:`` UNIONS
# with the policy; per-source ``exempt_dirs:`` SUBTRACTS from it. The
# resolved per-source set is computed in ``Config.resolve_excluded_dirs``.
#
# Anything added here ships globally — keep entries to "essentially
# never useful to index" cases (build artifacts, caches, vendored deps,
# IDE / VCS metadata).
DEFAULT_EXCLUDE_POLICY: tuple[str, ...] = (
    # VCS metadata
    '.git', '.hg', '.svn',
    # Python venvs / caches / build
    '.venv', 'venv', 'env', '.env',
    '__pycache__', 'site-packages', '.eggs',
    '.mypy_cache', '.ty_cache', '.pytest_cache', '.ruff_cache',
    '.tox', '.nox',
    '.hypothesis',
    # JS / TS package managers + vendored ESM bundles
    'node_modules', 'web_modules',
    'bower_components', 'jspm_packages',
    # JS / TS framework dev caches and build outputs
    '.next', '.nuxt', '.svelte-kit',
    '.cache', '.parcel-cache', '.turbo',
    '.vite', '.angular',
    '.docusaurus', '.umi', 'storybook-static',
    '__generated__',
    # JS / TS deploy / hosting platform caches
    '.vercel', '.netlify',
    # JVM / Scala build + tooling
    'target', '.gradle', '.bsp', '.metals', '.bloop',
    '.scala-build',
    # Generic build output
    'build', 'dist', 'out',
    # Test / coverage output
    'coverage', 'htmlcov', '.coverage',
    # Infra-as-code / cloud build caches (provider downloads, plan
    # binaries, synthesized templates — never useful to index)
    '.terraform', '.pulumi', '.serverless', '.aws-sam',
    '.firebase', '.cdk.out',
    # Generated docs / static sites
    '_site', 'generated-docs', '_build',
    # IDE / editor metadata
    '.idea', '.vscode', '.vs', '.fleet',
    # Vendored deps (Go, PHP, ...)
    'vendor',
    # CI / VCS-platform config (workflows, hooks, release tooling)
    '.github', '.gitlab', '.circleci', '.husky', '.changeset',
    # Transient / runtime output
    'tmp', 'logs',
    # Ariadne's own output dir (SCIP index, manifest, intermediate maps).
    # Never index ourselves — "don't scan yourself" is a built-in default.
    '.ariadne',
)


# Companion to DEFAULT_EXCLUDE_POLICY (directories): default GLOB patterns
# matched per-file via ``Path.match``, unioned with a source's ``exclude``.
# These are the pre-existing test-scaffolding globs (kept verbatim — tests
# themselves may warrant docs, so this is unchanged behaviour, not a claim
# that test code is never documented) PLUS vendored third-party utilities
# that are not project source and never will be: pip / poetry bootstrap
# installers dropped at a repo root (e.g. a 2.6 MB get-pip.py that would
# otherwise dominate the cost estimate and fail generation).
DEFAULT_EXCLUDE_FILE_PATTERNS: tuple[str, ...] = (
    # Test scaffolding (pre-existing default — unchanged)
    '**/test_*.py', '**/*_test.py', '**/conftest.py',
    # Vendored third-party installers — not project source
    '**/get-pip.py', '**/get-poetry.py', '**/install-poetry.py',
    # macOS Finder metadata
    '**/.DS_Store',
)


@frozen
class SourceConfig:
    """Configuration for a documentation source.

    Attributes:
        path: Path to the source code directory.
        depends_on: List of source names this source depends on.
        parent: Name of parent source (for subdirectory sources).
        branches: List of git branch patterns where this source is active.
            If empty or not set, source is active on all branches.
            Supports glob patterns (e.g., "feature/*", "develop").
        ref: Pin external dependency to specific branch/tag.
        exclude: Glob patterns matched against ``Path.match`` to exclude
            files from generation/sync. Use this for files that ARE in a
            catalog-eligible extension but contain secrets you don't want
            sent to LLMs / embeddings (e.g. ``"**/.env*"``,
            ``"**/secrets/**"``, ``"**/credentials.json"``).
        exclude_dirs: Directory names to ADD to the global
            ``exclude_policy`` for this source.
        exempt_dirs: Directory names to REMOVE from the global
            ``exclude_policy`` for this source — opt-in to walking a
            dir that the policy would normally skip (e.g. a project
            that legitimately contains a ``dist/`` worth indexing).
    """

    path: str | None = None
    depends_on: tuple[str, ...] = ()
    parent: str | None = None
    branches: tuple[str, ...] = ()
    ref: str | None = None
    exclude: tuple[str, ...] = ()
    # Directory NAMES (not patterns) to prune from the discovery walk
    # at every depth. More efficient than glob exclusion since it skips
    # entire subtrees. Use for "exclude this whole tree" cases like
    # ``docs``, ``generated``, ``target``, ``vendor``.
    exclude_dirs: tuple[str, ...] = ()
    # Directory NAMES exempt from the global exclude_policy for this
    # source. Lets a project opt back into walking a dir that the
    # policy would normally skip. Empty default — most projects never
    # touch this. Validated against ``exclude_dirs`` at config load.
    exempt_dirs: tuple[str, ...] = ()
    # Free-form environment hints for indexer setup (Phase 2e). Keys
    # discovery / ariadne index recognize today: ``python_path`` (path
    # to a Python interpreter for scip-python). Will grow over time
    # (e.g., ``node_path``, ``jdk_path``) without schema migrations.
    # Per design decision #6, this is the only SCIP-related field
    # users may need to set in ariadne.yaml — everything else is
    # auto-derived by ``ariadne discover``.
    env_hints: dict[str, str] = field(factory=dict)
    # OpenAPI/Swagger spec paths (relative to source path) — Phase 7b.
    # When non-empty, ariadne sync (Wave 4) parses these as
    # authoritative producer-side endpoint declarations and binds
    # operationIds back to scip_symbols. Empty default — sources
    # without published specs continue to use pattern-based extraction
    # (Phase 8) where it applies.
    swagger_paths: tuple[str, ...] = ()
    ignore_staleness: bool | tuple[str, ...] = False

# Config file names to search for
CONFIG_FILENAME = 'ariadne.yaml'

# config.py lives at the repo root, so its own directory is where a
# project-local ariadne.yaml sits. Used as a fallback config location when the
# process cwd isn't the project — e.g. ``ariadne mcp`` launched via
# ``uv run --directory <repo>``, which does NOT chdir the spawned child, so a
# cwd-only search misses. Lets the server resolve its config with no
# ARIADNE_CONFIG and no manual setup.
_PACKAGE_ROOT = Path(__file__).resolve().parent


def config_search_paths() -> list[Path]:
    """Ordered locations searched for ``ariadne.yaml``.

    ``ARIADNE_CONFIG`` (if set) → cwd → package/repo root → home. The
    package-root rung lets a process launched outside the project tree (e.g.
    the MCP server via ``uv run --directory``) still resolve its config.
    """
    paths: list[Path] = []
    env_config = os.environ.get('ARIADNE_CONFIG')
    if env_config:
        paths.append(Path(env_config))
    paths.extend([
        Path.cwd() / CONFIG_FILENAME,
        _PACKAGE_ROOT / CONFIG_FILENAME,
        Path.home() / CONFIG_FILENAME,
    ])
    return paths

# Default configuration values
_DEFAULT_MENTION_MESSAGE = (
    "When Ariadne documentation helps you answer a question or complete a task, "
    "casually mention it (e.g., 'Consulted Ariadne and found that...', "
    "'Ariadne's docs on X clarified...'). Keep it natural — don't force it."
)

DEFAULTS = {
    'model': 'gpt-5.5',
    'provider': 'openai',
    'db_path': 'ariadne.db',
    'docs_base': './docs',
    'default_source': None,
    'sources': {},
    'main_branch': 'main',
    'branch_doc_ttl_days': 180,
    'mention_ariadne': {'enabled': True, 'message': _DEFAULT_MENTION_MESSAGE},
    'response_token_budget': 20000,
    'themes_enabled': True,
    'exclude_policy': DEFAULT_EXCLUDE_POLICY,
}


class Config:
    """Configuration manager for Ariadne.

    Loads configuration from ariadne.yaml, searching in:
    1. Current working directory
    2. User's home directory (~/)

    Provides source path resolution and sensible defaults.
    """

    def __init__(self, config_path: Path | None = None):
        """Initialize config from file or defaults.

        Args:
            config_path: Explicit path to config file. If None, searches
                        standard locations (cwd, then home dir).
        """
        self._config: dict[str, Any] = dict(DEFAULTS)
        self._config_path: Path | None = None

        if config_path:
            self._load_from_path(config_path)
        else:
            self._load_from_standard_locations()

    def _load_from_standard_locations(self) -> None:
        """Search for and load config from standard locations.

        Order: ``ARIADNE_CONFIG`` → CWD → package/repo root → home (see
        :func:`config_search_paths`). The package-root rung lets a process
        launched outside the project tree (e.g. the MCP server via
        ``uv run --directory``, which doesn't chdir the child) still resolve
        its config.
        """
        for path in config_search_paths():
            if path.exists():
                self._load_from_path(path)
                return

    def _load_from_path(self, path: Path) -> None:
        """Load configuration from a specific file path.

        Raises:
            ConfigError: If the loaded config violates a hard invariant
                (currently: a source declaring the same dir in both
                ``exclude_dirs`` and ``exempt_dirs``). I/O / parse errors
                fall back to defaults silently as before.
        """
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            # Silently fall back to defaults on error.
            return

        self._config_path = Path(path) if not isinstance(path, Path) else path

        # Merge loaded config with defaults
        for key, value in data.items():
            if key == 'defaults':
                # Handle nested defaults section
                for dk, dv in value.items():
                    if dk in DEFAULTS:
                        self._config[dk] = dv
            else:
                self._config[key] = value

        self._validate()

    def _validate(self) -> None:
        """Run hard-fail validations on the loaded config.

        Per-source checks:
        - Unknown keys raise ConfigError with a difflib suggestion
          (e.g., ``purh: /foo`` → "did you mean 'path'?"). Without
          this guard, typos silently produce a SourceConfig with
          empty ``path``, and the downstream walk indexes whatever
          directory cwd happens to resolve to — a multi-thousand-file
          surprise.
        - ``path`` must be present and non-empty. Same rationale.
        - ``exclude_dirs`` and ``exempt_dirs`` must not overlap (a dir
          cannot be both skipped and force-walked).
        - ``env_hints`` must be a mapping of strings to strings.
          Misconfigured values surface immediately at config load
          rather than later at indexer-invocation time.
        """
        sources = self._config.get('sources', {}) or {}
        for source_name, raw in sources.items():
            if not isinstance(raw, dict):
                continue

            # Unknown-key check first — when the user typo'd a known
            # key, this gives the most actionable error (names the
            # typo + suggests the intended key). The missing-path
            # check below would fire too in that case but with a
            # less informative message.
            #
            # cutoff=0.5 catches the canonical motivating case
            # ``purh: /foo`` → suggest ``path`` (ratio is exactly
            # 0.5: matches 'p' and 'h', 2*2/8). Higher cutoffs miss
            # this; lower cutoffs start to suggest unrelated keys.
            for key in raw:
                if key not in _SOURCE_KNOWN_KEYS:
                    suggestions = difflib.get_close_matches(
                        key, _SOURCE_KNOWN_KEYS, n=1, cutoff=0.5,
                    )
                    hint = (
                        f" — did you mean '{suggestions[0]}'?"
                        if suggestions else ''
                    )
                    raise ConfigError(
                        f"Source '{source_name}': unknown key "
                        f"'{key}'{hint}",
                    )

            # Required-field check: ``path`` must be present and
            # non-empty. Empty/missing paths resolve to cwd via
            # Path('').resolve(), which is the silent-walk footgun
            # this validation exists to prevent.
            path_value = raw.get('path')

            excl = set(raw.get('exclude_dirs', []) or [])
            exempt = set(raw.get('exempt_dirs', []) or [])
            overlap = excl & exempt
            if overlap:
                raise ConfigError(
                    f"Source '{source_name}': directory name(s) "
                    f"{sorted(overlap)} appear in both 'exclude_dirs' and "
                    f"'exempt_dirs'. A directory cannot be both skipped "
                    f"and force-walked — remove from one of the lists."
                )

            env_hints = raw.get('env_hints')
            if env_hints is not None:
                if not isinstance(env_hints, dict):
                    raise ConfigError(
                        f"Source '{source_name}': 'env_hints' must be a "
                        f'mapping (e.g., ``env_hints: {{python_path: '
                        f'/path/to/python}}``), got '
                        f'{type(env_hints).__name__}.'
                    )
                for k, v in env_hints.items():
                    if not isinstance(k, str) or not isinstance(v, str):
                        raise ConfigError(
                            f"Source '{source_name}': 'env_hints' must be "
                            f'a mapping of strings to strings; got '
                            f'{type(k).__name__}={type(v).__name__!r}.'
                        )

            swagger_paths = raw.get('swagger_paths')
            if swagger_paths is not None:
                if not isinstance(swagger_paths, list):
                    raise ConfigError(
                        f"Source '{source_name}': 'swagger_paths' must be "
                        f'a list of strings (e.g., ``swagger_paths: '
                        f'[api/openapi.yaml]``), got '
                        f'{type(swagger_paths).__name__}.'
                    )
                for sp in swagger_paths:
                    if not isinstance(sp, str):
                        raise ConfigError(
                            f"Source '{source_name}': 'swagger_paths' must "
                            f'contain only strings; got '
                            f'{type(sp).__name__}.'
                        )
            ign = raw.get('ignore_staleness')
            if ign is not None and not isinstance(ign, bool):
                if not isinstance(ign, list):
                    raise ConfigError(
                        f"Source '{source_name}': 'ignore_staleness' must be "
                        f'true/false or a list of glob patterns, got '
                        f'{type(ign).__name__}.'
                    )
                for pat in ign:
                    if not isinstance(pat, str):
                        raise ConfigError(
                            f"Source '{source_name}': 'ignore_staleness' globs "
                            f'must be strings; got {type(pat).__name__}.'
                        )

    @property
    def config_path(self) -> Path | None:
        """Return the path to the loaded config file, if any."""
        return self._config_path

    @property
    def config_dir(self) -> Path:
        """Return the directory containing the config file, or cwd as fallback."""
        if self._config_path:
            return self._config_path.parent
        return Path.cwd()

    def _resolve_config_path(self, key: str, default: str) -> Path:
        """Resolve a config path relative to the config file directory."""
        path = Path(self._config.get(key, default))
        if not path.is_absolute():
            path = self.config_dir / path
        return path.resolve()

    def _from_defaults(self, key: str, fallback):
        """Look up ``key`` first at top-level, then under the
        optional ``defaults:`` section, then fall back.

        Lets users group LLM-related settings under one ``defaults:``
        block (the canonical layout the property docstrings promise)
        without breaking older configs that put the same keys at
        top-level. Top-level wins on conflict — explicit override
        beats the defaults block.
        """
        if key in self._config:
            return self._config[key]
        defaults_section = self._config.get('defaults', {}) or {}
        if isinstance(defaults_section, dict) and key in defaults_section:
            return defaults_section[key]
        return fallback

    @property
    def model(self) -> str:
        """Return the default LLM model.

        Reads ``model:`` from top-level first, then from the optional
        ``defaults:`` section in ``ariadne.yaml``.
        """
        return self._from_defaults('model', DEFAULTS['model'])

    @property
    def provider(self) -> str:
        """LLM backend selector — ``"openai"`` or ``"anthropic"``.

        Set via the ``provider:`` key under ``defaults:`` in
        ``ariadne.yaml``, or top-level. Top-level wins on conflict.
        Defaults to ``"openai"`` for backwards-compat.
        """
        return self._from_defaults('provider', 'openai')

    @property
    def db_path(self) -> str:
        """Return the default database path, resolved relative to config file."""
        return str(self._resolve_config_path('db_path', DEFAULTS['db_path']))

    @property
    def staleness_db_path(self) -> str:
        """Return the staleness tracking database path, resolved relative to config file."""
        return str(self._resolve_config_path('staleness_db_path', 'ariadne_staleness.db'))

    @property
    def default_source(self) -> str | None:
        """Return the default source name."""
        return self._config.get('default_source')

    @property
    def sources(self) -> dict[str, str | dict]:
        """Return the sources mapping (name -> path or config dict)."""
        return self._config.get('sources', {})

    def get_source_config(self, source_name: str) -> SourceConfig | None:
        """Get the SourceConfig for a named source.

    Returns None only when the source is not configured at all. A configured
    source with no ``path`` (a serve-only source whose docs live in the DB)
    yields a SourceConfig with ``path=None`` — distinct from None, which means
    "unknown source".
    """
        if source_name not in self.sources:
            return None
        raw = self.sources.get(source_name)
        if raw is None:
            return SourceConfig(path=None)
        if isinstance(raw, str):
            return SourceConfig(path=raw)
        if isinstance(raw, dict):
            return SourceConfig(
                path=raw.get('path'),
                depends_on=tuple(raw.get('depends_on', [])),
                parent=raw.get('parent'),
                branches=tuple(raw.get('branches', [])),
                ref=raw.get('ref'),
                exclude=tuple(raw.get('exclude', [])),
                exclude_dirs=tuple(raw.get('exclude_dirs', [])),
                exempt_dirs=tuple(raw.get('exempt_dirs', [])),
                env_hints=dict(raw.get('env_hints') or {}),
                swagger_paths=tuple(raw.get('swagger_paths', []) or ()),
            ignore_staleness=_coerce_ignore_staleness(raw.get('ignore_staleness', False)))
        return None

    def hydrate_relations(self, all_relations: dict) -> None:
        """Layer the DB-persisted source graph onto the yaml config.

        For every source — those declared in yaml and those known only to the
        DB — fill each relational field (``depends_on`` / ``parent`` /
        ``branches``) from ``all_relations`` WHEN yaml omits it
        (yaml-when-present-else-DB, per field), and ADD sources present only in
        the DB. This lets a serving box resolve the full closure from
        ``ariadne.db`` without restating the graph in its own ariadne.yaml.
        Idempotent; a no-op when ``all_relations`` is empty.
        """
        sources = self._config.get('sources') or {}
        hydrated: dict = {}
        for name in set(sources) | set(all_relations):
            raw = sources.get(name)
            if raw is None:
                entry: dict = {}
            elif isinstance(raw, str):
                entry = {'path': raw}
            else:
                entry = dict(raw)
            db = all_relations.get(name, {})
            for field in ('depends_on', 'parent', 'branches'):
                if field not in entry and field in db:
                    entry[field] = db[field]
            hydrated[name] = entry
        self._config['sources'] = hydrated

    def get_source_path(self, source_name: str) -> Path | None:
        """Get the resolved path for a named source, or None.

    None when the source is unknown OR has no ``path`` (serve-only). Callers
    that need a real path (generation, indexing) must treat None as "not
    buildable" and fail loud rather than walking cwd.
    """
        config = self.get_source_config(source_name)
        if config is None or not config.path:
            return None
        return Path(config.path).expanduser().resolve()

    def get_source_scip_config(self, source_name: str):
        """Return the SCIP configuration for a source, or None if not configured.

        SCIP-backed extraction (Scala/Java via SemanticDB) requires both an
        ``index_kinds`` mapping AND a ``scip:`` block under the source. If
        either is absent we return ``None``; the caller falls back to
        ast-grep for those file types unless ``index_kinds`` declares SCIP
        explicitly (in which case absence becomes a fail-loud error
        downstream — see ``docgen.scip_config.ScipError``).
        """
        from docgen.scip_config import SourceScipConfig

        raw = self.sources.get(source_name)
        if not isinstance(raw, dict):
            return None
        scip = raw.get('scip')
        if not isinstance(scip, dict):
            return None
        artifact = scip.get('artifact_path')
        if not artifact:
            return None
        return SourceScipConfig(
            repo=source_name,
            artifact_path=Path(artifact),
            max_staleness_days=int(scip.get('max_staleness_days', 7)),
            index_kinds=dict(raw.get('index_kinds', {})),
            allow_degraded=False,
            vue_mappings=self._load_vue_mappings(Path(artifact)),
        )

    @staticmethod
    def _load_vue_mappings(artifact_path: Path) -> dict:
        """Collect and merge every ``vue_mapping`` JSON declared by the
        source's manifest (sibling to the merged ``.scip`` artifact).

        Returns ``{companion_rel_path: {original, line_offset, ...}}`` so
        ``resolve_index`` can translate companion paths back to ``.vue``.
        Empty dict when there's no manifest or no Vue entries — the common
        (non-Vue) case pays only one stat + read.
        """
        import json

        manifest_path = artifact_path.parent / 'manifest.json'
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return {}
        ariadne_dir = artifact_path.parent
        merged: dict = {}
        for entry in manifest.get('indexers', ()):
            rel = entry.get('vue_mapping')
            if not rel:
                continue
            try:
                data = json.loads(
                    (ariadne_dir / rel).read_text(encoding='utf-8'),
                )
            except (OSError, ValueError):
                continue
            merged.update(data)
        return merged

    def get_source_dependencies(self, source_name: str) -> list[str]:
        """Get the list of dependency source names for a source.

        Args:
            source_name: The source name to look up.

        Returns:
            List of source names this source depends on.
        """
        config = self.get_source_config(source_name)
        if config is None:
            return []
        return list(config.depends_on)

    def scope_closure(self, source_name: str) -> frozenset[str]:
        """Directional closure of ``source_name`` in the ``depends_on`` graph.

        Forward closure for a source that declares ``depends_on`` — the
        source plus everything it transitively depends on. Reverse closure
        for a leaf (no declared deps) — the source plus everything that
        transitively depends on it. This lets a shared library like an
        auth service see its consumer-side context naturally.

        Phase 1 of the closure-scoping design (see
        ``designs/directional-closure-scoping.md``).
        """
        if source_name not in self.sources:
            configured = sorted(self.sources)
            raise KeyError(
                f'unknown source {source_name!r}; configured sources: '
                f'{configured}',
            )
        if self.get_source_dependencies(source_name):
            return self._forward_closure(source_name)
        return self._reverse_closure(source_name)

    def _forward_closure(self, source_name: str) -> frozenset[str]:
        self._reject_cycles_in_forward_deps(source_name)
        visited: set[str] = set()
        stack: list[str] = [source_name]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(self.get_source_dependencies(current))
        return frozenset(visited)

    def _reverse_closure(self, source_name: str) -> frozenset[str]:
        # Invert the dep graph once across all configured sources.
        reverse: dict[str, list[str]] = {}
        for name in self.sources:
            for dep in self.get_source_dependencies(name):
                reverse.setdefault(dep, []).append(name)

        visited: set[str] = set()
        stack: list[str] = [source_name]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(reverse.get(current, ()))
        return frozenset(visited)

    def _reject_cycles_in_forward_deps(self, source_name: str) -> None:
        """DFS the forward `depends_on` graph and raise if any cycle is
        reachable from `source_name`. Tracks the current path so the
        error message can name the offending nodes.
        """
        on_path: list[str] = []
        on_path_set: set[str] = set()
        completed: set[str] = set()

        def visit(node: str) -> None:
            if node in on_path_set:
                cycle = on_path[on_path.index(node):] + [node]
                raise ValueError(
                    f'cycle detected in depends_on: '
                    f'{" -> ".join(cycle)}',
                )
            if node in completed:
                return
            on_path.append(node)
            on_path_set.add(node)
            for dep in self.get_source_dependencies(node):
                visit(dep)
            on_path.pop()
            on_path_set.remove(node)
            completed.add(node)

        visit(source_name)

    def _mutate_config(self, mutate) -> bool:
        """Read ariadne.yaml, apply ``mutate(data)`` to the parsed dict in
        place, write it back, and reload the live config.

        ``mutate`` may return ``False`` to abort the write (e.g. the
        target source doesn't exist) — the file is then left untouched and
        this returns ``False``. Any other return value (typically
        ``None``) commits the write. Returns ``False`` when there's no
        config path or the file can't be read/written.
        """
        if self._config_path is None:
            return False
        try:
            with open(self._config_path) as f:
                data = yaml.safe_load(f) or {}
            if mutate(data) is False:
                return False
            with open(self._config_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            self._load_from_path(self._config_path)
            return True
        except (OSError, yaml.YAMLError):
            return False

    def set_source_dependencies(self, source_name: str, deps: list[str]) -> bool:
        """Persist dependency configuration for a source to ariadne.yaml.

        Args:
            source_name: The source name to update.
            deps: List of source names this source depends on.

        Returns:
            True if successfully saved, False otherwise.
        """
        def mutate(data):
            sources = data.setdefault('sources', {})
            current = sources.get(source_name)
            if current is None:
                return False
            if isinstance(current, str):
                # Convert simple path to full config
                sources[source_name] = {'path': current, 'depends_on': deps}
            elif isinstance(current, dict):
                current['depends_on'] = deps
            else:
                return False

        return self._mutate_config(mutate)

    def set_source_config(
        self,
        source_name: str,
        *,
        path: str | None = None,
        depends_on: list[str] | None = None,
        parent: str | None = None,
        branches: list[str] | None = None,
        ref: str | None = None,
        exclude: list[str] | None = None,
        exclude_dirs: list[str] | None = None,
    ignore_staleness: bool | None = None) -> bool:
        """Persist full source configuration to ariadne.yaml.

        Creates the ``sources.<name>`` entry if it does not exist, else
        updates only the fields that are provided (``None`` leaves the
        existing value untouched), so repeated calls are idempotent.

        Args:
            source_name: The source name to update or create.
            path: Path to the source code directory.
            depends_on: List of source names this source depends on.
            parent: Name of parent source.
            branches: List of git branch patterns.
            ref: Pin to specific branch/tag.
            exclude: Glob patterns of files to exclude.
            exclude_dirs: Directory names to exclude.

        Returns:
            True if successfully saved, False otherwise.
        """
        def mutate(data):
            sources = data.setdefault('sources', {})
            current = sources.get(source_name, {})
            if isinstance(current, str):
                current = {'path': current}
            # Update fields if provided
            if path is not None:
                current['path'] = path
            if depends_on is not None:
                current['depends_on'] = depends_on
            if parent is not None:
                current['parent'] = parent
            if branches is not None:
                current['branches'] = branches
            if ref is not None:
                current['ref'] = ref
            if exclude is not None:
                current['exclude'] = exclude
            if exclude_dirs is not None:
                current['exclude_dirs'] = exclude_dirs
            if ignore_staleness is not None:
                current['ignore_staleness'] = ignore_staleness
            sources[source_name] = current

        return self._mutate_config(mutate)

    def remove_source(self, source_name: str) -> bool:
        """Delete the ``sources.<name>`` entry from ariadne.yaml.

        Also clears ``default_source`` if it pointed at the removed
        source, so the config never references a source that no longer
        exists.

        Returns:
            True if the source existed and was removed, False if it was
            absent or the file could not be written.
        """
        def mutate(data):
            sources = data.get('sources')
            if not isinstance(sources, dict) or source_name not in sources:
                return False
            del sources[source_name]
            if data.get('default_source') == source_name:
                data['default_source'] = None

        return self._mutate_config(mutate)

    def set_default_source(self, source_name: str) -> bool:
        """Persist ``default_source`` to ariadne.yaml."""
        def mutate(data):
            data['default_source'] = source_name

        return self._mutate_config(mutate)

    @property
    def docs_base(self) -> Path:
        """Return the base directory for documentation output, resolved relative to config file."""
        return self._resolve_config_path('docs_base', DEFAULTS['docs_base'])

    @property
    def main_branch(self) -> str:
        """Return the main branch name for comparison."""
        return self._config.get('main_branch', DEFAULTS['main_branch'])

    @property
    def response_token_budget(self) -> int:                                                                                                                             
        """Max tokens in an ariadne_search response before sections-downgrade + truncation."""
        return self._config.get('response_token_budget', DEFAULTS['response_token_budget'])                                                                             
                                                                                                                                                                        
    @property
    def branch_doc_ttl_days(self) -> int:
        """Return the TTL in days for branch-specific documents."""
        return self._config.get('branch_doc_ttl_days', DEFAULTS['branch_doc_ttl_days'])

    @property
    def themes_enabled(self) -> bool:
        """Master switch for the cross-cutting themes pipeline (Themes plan §7)."""
        return bool(self._config.get('themes_enabled', DEFAULTS['themes_enabled']))

    @property
    def exclude_policy(self) -> tuple[str, ...]:
        """Globally excluded directory names — REPLACES the default when set.

        Returns ``DEFAULT_EXCLUDE_POLICY`` unless the user defined a
        top-level ``exclude_policy:`` in ``ariadne.yaml``. ``[]`` is
        honored verbatim (full walk).
        """
        raw = self._config.get('exclude_policy', DEFAULT_EXCLUDE_POLICY)
        return tuple(raw) if raw is not None else ()

    def resolve_excluded_dirs(self, source_name: str | None) -> tuple[str, ...]:
        """Compute the effective excluded-dir set for a source.

        Formula::

            (exclude_policy ∪ source.exclude_dirs) − source.exempt_dirs

        Returns a sorted tuple for determinism. ``source_name`` may be
        ``None`` or a name not in ``sources`` — in either case the
        global policy is returned unmodified.
        """
        policy = set(self.exclude_policy)
        if source_name is None:
            return tuple(sorted(policy))
        sc = self.get_source_config(source_name)
        if sc is None:
            return tuple(sorted(policy))
        effective = (policy | set(sc.exclude_dirs)) - set(sc.exempt_dirs)
        return tuple(sorted(effective))

    @property
    def mention_ariadne_enabled(self) -> bool:
        """Return whether the 'mention Ariadne' behavioral directive is enabled."""
        raw = self._config.get('mention_ariadne', {})
        if isinstance(raw, dict):
            return bool(raw.get('enabled', True))
        return bool(raw)

    @property
    def mention_ariadne_message(self) -> str:
        """Return the behavioral directive message text."""
        raw = self._config.get('mention_ariadne', {})
        if isinstance(raw, dict):
            return str(raw.get('message', _DEFAULT_MENTION_MESSAGE))
        return _DEFAULT_MENTION_MESSAGE

    def resolve_docs_path(self, source: str) -> Path:
        """Resolve the documentation output path for a source.

        Args:
            source: The source name (e.g., 'mylib').

        Returns:
            Path to the docs directory for this source (e.g., ./docs/mylib/).
        """
        return self.docs_base / source

    def resolve_source(self, source: str | None) -> Path | None:
        """Resolve a source name or path to an absolute Path.

    For a configured source name, return its resolved path — or None if it is
    serve-only (no ``path``). We do NOT fall through to treating the name as a
    filesystem path, which would silently resolve to cwd/<name>. A
    non-configured argument is treated as a path.
    """
        if source is None:
            source = self.default_source
        if source is None:
            return None
        if source in self.sources:
            return self.get_source_path(source)
        path = Path(source).expanduser()
        if path.exists():
            return path.resolve()
        return path.resolve()

    def get_all_source_paths(self) -> dict[str, Path]:
        """Get all configured source names mapped to their resolved paths.

        Returns:
            Dict mapping source names to their resolved paths.
        """
        result = {}
        for name in self.sources:
            path = self.get_source_path(name)
            if path:
                result[name] = path
        return result

    def is_source_active(self, source_name: str, branch: str | None) -> bool:
        """Check if a source is active for a given branch.

        A source is active if:
        - It has no branches restriction (empty list = active on all branches)
        - Its branches list includes "*" (wildcard for all branches)
        - The current branch matches one of the branch patterns

        Args:
            source_name: The source name to check.
            branch: Current git branch (None means ignore branch filtering).

        Returns:
            True if the source is active for the given branch.
        """
        import fnmatch

        config = self.get_source_config(source_name)
        if config is None:
            return False

        # No branch restriction = active on all branches
        if not config.branches:
            return True

        # If branch is unknown, only include sources without restrictions
        if branch is None:
            return not config.branches

        # Check if branch matches any pattern
        for pattern in config.branches:
            if pattern == '*' or fnmatch.fnmatch(branch, pattern):
                return True

        return False

    def get_source_scope(self, cwd: Path, branch: str | None = None) -> str | None:
        """Resolve which source applies based on working directory and branch.

        Finds the most specific source whose path contains the cwd and is
        active on the current branch.

        Args:
            cwd: Current working directory.
            branch: Current git branch (None to ignore branch filtering).

        Returns:
            Source name if found, None otherwise.
        """
        cwd_resolved = cwd.resolve()
        candidates: list[tuple[str, Path]] = []

        for name in self.sources:
            if not self.is_source_active(name, branch):
                continue

            source_path = self.get_source_path(name)
            if source_path is None:
                continue

            # Check if cwd is within this source's path
            try:
                cwd_resolved.relative_to(source_path)
                candidates.append((name, source_path))
            except ValueError:
                # cwd is not under this source path
                continue

        if not candidates:
            return None

        # Return the most specific (longest path) match
        candidates.sort(key=lambda x: len(x[1].parts), reverse=True)
        return candidates[0][0]

    def get_effective_dependencies(
        self,
        source_name: str,
        branch: str | None = None,
    ) -> list[str]:
        """Get all effective dependencies for a source.

        This includes:
        - The parent source (if any)
        - Explicitly declared dependencies
        - Dependencies filtered by branch patterns

        Args:
            source_name: The source name to get dependencies for.
            branch: Current git branch (for filtering branch-specific deps).

        Returns:
            List of source names this source effectively depends on.
        """
        config = self.get_source_config(source_name)
        if config is None:
            return []

        deps: list[str] = []

        # Add parent as implicit dependency (if parent is active)
        if config.parent and self.is_source_active(config.parent, branch):
            deps.append(config.parent)

        # Add explicit dependencies (if they are active on current branch)
        for dep in config.depends_on:
            if dep not in deps and self.is_source_active(dep, branch):
                deps.append(dep)

        return deps

    def validate_source_hierarchy(self) -> list[str]:
        """Validate source hierarchy relationships.

        Checks that:
        - Parent sources exist
        - Child paths are subdirectories of parent paths
        - No circular parent relationships

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []

        for name in self.sources:
            config = self.get_source_config(name)
            if config is None or config.parent is None:
                continue

            # Check parent exists
            parent_config = self.get_source_config(config.parent)
            if parent_config is None:
                errors.append(
                    f"Source '{name}' has parent '{config.parent}' which does not exist"
                )
                continue

            # Check path hierarchy
            child_path = self.get_source_path(name)
            parent_path = self.get_source_path(config.parent)

            if child_path is None or parent_path is None:
                continue

            try:
                child_path.relative_to(parent_path)
            except ValueError:
                errors.append(
                    f"Source '{name}' path '{child_path}' is not a subdirectory "
                    f"of parent '{config.parent}' path '{parent_path}'"
                )

        # Check for circular parent relationships
        for name in self.sources:
            visited: set[str] = set()
            current = name
            while current:
                if current in visited:
                    errors.append(f"Circular parent relationship detected involving '{name}'")
                    break
                visited.add(current)
                config = self.get_source_config(current)
                current = config.parent if config else None

        return errors

    def get_sources_for_branch(self, branch: str | None) -> list[str]:
        """Get all source names active for a given branch.

        Args:
            branch: Current git branch (None = only base sources).

        Returns:
            List of active source names.
        """
        return [
            name for name in self.sources
            if self.is_source_active(name, branch)
        ]

    def to_dict(self) -> dict[str, Any]:
        """Return the full configuration as a dictionary."""
        return dict(self._config)
    def source_staleness_exempt(self, source_name: str) -> bool:
        """True iff the whole source opts out of staleness checks
    (``ignore_staleness: true``). Source-level gates such as the SCIP
    index-age check consult this; a file-glob exemption does not count.
    """
        sc = self.get_source_config(source_name)
        return bool(sc is not None and sc.ignore_staleness is True)
    def path_staleness_exempt(self, source_name: str, rel_path: str) -> bool:
        """True iff staleness should be ignored for ``rel_path`` within this
    source -- either the whole source is exempt (``ignore_staleness:
    true``) or ``rel_path`` matches one of the configured exemption globs.
    """
        sc = self.get_source_config(source_name)
        if sc is None:
            return False
        return ignore_staleness_matches(sc.ignore_staleness, rel_path)
    def effective_scip_staleness_days(self, source_name: str) -> int | None:
        """Max age in days allowed for this source's SCIP index, or ``None``
    to disable the age gate when the source is staleness-exempt
    (``ignore_staleness: true``); otherwise the configured
    ``scip.max_staleness_days`` (default 7).
    """
        if self.source_staleness_exempt(source_name):
            return None
        scip = self.get_source_scip_config(source_name)
        return scip.max_staleness_days if scip else 7
    def source_ignore_staleness(self, source_name):
        """The raw ``ignore_staleness`` value the CLI passes to
    OrchestratorConfig: ``True`` / a glob tuple / ``False`` (also False
    for an unknown or unset source).
    """
        sc = self.get_source_config(source_name) if source_name else None
        return sc.ignore_staleness if sc is not None else False


# Global config instance - loaded lazily
_global_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance.

    Creates and loads config on first access.
    """
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def reload_config() -> Config:
    """Force reload of the global configuration."""
    global _global_config
    _global_config = Config()
    return _global_config
def _coerce_ignore_staleness(value):
    """Normalize a raw ``ignore_staleness`` value into the stored form:
    ``True`` (whole source exempt), a tuple of glob patterns (specific
    files exempt), or ``False`` (default — staleness checks apply).
    """
    if value is True:
        return True
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return False
def ignore_staleness_matches(value, rel_path) -> bool:
    """True iff ``value`` -- a ``SourceConfig.ignore_staleness`` (``True``
    for the whole source, a tuple of globs for specific files, or
    ``False``) -- marks ``rel_path`` as staleness-exempt. ``fnmatch``
    semantics, so ``vendor/**`` covers everything under ``vendor/``.
    """
    if value is True:
        return True
    if not value:
        return False
    from fnmatch import fnmatch
    rel = str(rel_path).replace('\\', '/')
    return any(fnmatch(rel, pat) for pat in value)
