"""Spool enablement + manifest gating — slice (a) of the Spool plugin.

Design: designs/spool-environment-plugin.md §9 (manifest schema) ·
§18.2 (runtime pin, fail-closed) · §18.6.4 (config-static enablement via
the ``spools:`` mapping). Enable-side only: config, manifest, gating,
registration; pack *import* is slice (b).
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from config import OFFICIAL_DOC_PROVENANCE


class SpoolError(Exception):
    """A Spool configuration/manifest violation — always loud, never masked."""


def load_yaml_mapping(path, error_cls):
    """Read a YAML file that must be a mapping, or raise ``error_cls``.

    The one place the spool loaders (manifest, packfile, guardrail catalog,
    concern taxonomy) share their read+parse+shape idiom, so it can't drift.
    """
    try:
        data = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    except (OSError, yaml.YAMLError) as exc:
        raise error_cls(f'cannot read {path}: {exc}') from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise error_cls(f'{path} must be a mapping, got {type(data).__name__}')
    return data


_MANIFEST_REQUIRED_FIELDS = ('environment', 'version', 'target_runtime', 'checksum')

# CRIT-9: installed spool docs live under a RESERVED source id so a spool
# can never collide with (and pollute) a real configured source of the same
# name. The ``:`` makes it un-nameable via ``source add`` / ariadne.yaml,
# and the origin axis (scope + ranking) keys on this id, so disable cleanly
# drops it from scope and uninstall can target exactly these rows.
_SPOOL_SOURCE_PREFIX = 'spool:'


def spool_source_id(name: str) -> str:
    """The reserved store source id for an installed spool (CRIT-9)."""
    return f'{_SPOOL_SOURCE_PREFIX}{name}'


def is_spool_source(source_name) -> bool:
    """True iff ``source_name`` is a reserved spool source id (CRIT-9). The
    single definition of "this row belongs to a spool" — consumers use this
    instead of re-testing the prefix or reaching for the private constant."""
    return bool(source_name) and str(source_name).startswith(_SPOOL_SOURCE_PREFIX)


# --- Spool grounding gate (§18.1) -------------------------------------------
# A Spool must be indexable by a registered SCIP adapter. A corpus in a
# language with no adapter (e.g. Go) is refused up front rather than silently
# built from the low-confidence raw-file / ast-grep fallback — that path is
# exactly what a Spool exists to replace, so faking it defeats the point.

# Common source extensions we can NAME when reporting an ungrounded corpus.
_UNSUPPORTED_CODE_EXT_NAMES = {
    # NB: keep in sync with docgen.scip_languages — a language with a
    # registered indexer (e.g. Go via scip-go) must NOT appear here.
    '.rb': 'Ruby', '.rs': 'Rust', '.php': 'PHP', '.cs': 'C#',
    '.c': 'C', '.h': 'C', '.cpp': 'C++', '.cc': 'C++', '.hpp': 'C++',
    '.swift': 'Swift', '.m': 'Objective-C', '.mm': 'Objective-C',
    '.dart': 'Dart', '.ex': 'Elixir', '.exs': 'Elixir', '.clj': 'Clojure',
    '.hs': 'Haskell', '.erl': 'Erlang', '.lua': 'Lua', '.pl': 'Perl',
    '.zig': 'Zig',
}

_CORPUS_SCAN_SKIP_DIRS = frozenset({
    'node_modules', '.git', '.ariadne', '.venv', '__pycache__',
    'vendor', 'dist', 'build', 'target',
})


def is_scip_eligible(language) -> bool:
    """True iff ``language`` can be SCIP-indexed by a registered adapter —
    the bar for grounding a Spool (§18.1). Read from the SCIP language
    registry (``docgen.scip_languages``) so adding a new indexer (e.g.
    scip-go) flips its language eligible with no change here.
    """
    from docgen.scip_languages import _EXT_TO_LANG, LANGUAGES
    norm = str(language or '').strip().lower().lstrip('.')
    if not norm:
        return False
    if any(norm == lang.name or norm in lang.aliases for lang in LANGUAGES):
        return True
    return ('.' + norm) in _EXT_TO_LANG


def unsupported_corpus_language(root) -> str | None:
    """Scan a cloned corpus; if it contains source code but NONE a registered
    SCIP adapter can ground, return the dominant unsupported language's
    display name (so the caller can name it), else ``None``.

    ``None`` when any SCIP-indexable source is present (the pack is
    groundable — an incidental unsupported file is not a blocker) or when the
    corpus has no recognized code at all (a separate, out-of-scope case).
    """
    import os

    from docgen.scip_languages import _EXT_TO_LANG
    supported = 0
    unsupported: dict[str, int] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in _CORPUS_SCAN_SKIP_DIRS]
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in _EXT_TO_LANG:
                supported += 1
            elif ext in _UNSUPPORTED_CODE_EXT_NAMES:
                unsupported[ext] = unsupported.get(ext, 0) + 1
    if supported or not unsupported:
        return None
    dominant = max(unsupported, key=unsupported.get)
    return _UNSUPPORTED_CODE_EXT_NAMES[dominant]


# --------------------------------------------------------------------------
# License-admission gate (§18.1 redistribution safety). A spool ships DERIVED
# docs + a SCIP index built from its corpus, so the corpus must be under a
# license that permits redistributing derived work. Source-available /
# proprietary licenses (BUSL, SSPL, Elastic, PolyForm, Commons Clause, or an
# unrecognized/absent license) are NOT redistribution-safe for an open-source
# pack; the gate refuses them at build unless the builder opts into a
# local-only pack with --allow-nonfree.
# --------------------------------------------------------------------------

# Non-free markers are checked BEFORE open ones, so a restrictive rider (e.g.
# a Commons Clause over MIT, or BUSL over an otherwise-open base) dominates.
_NONFREE_LICENSE_MARKERS = (
    ('BUSL', ('business source license', 'busl-1', 'bsl-1')),
    ('SSPL', ('server side public license', 'sspl')),
    ('Elastic', ('elastic license',)),
    ('PolyForm', ('polyform',)),
    ('Commons Clause', ('commons clause',)),
)
_OPEN_LICENSE_MARKERS = (
    ('permissive', 'Apache-2.0', ('apache license',)),
    ('weak-copyleft', 'MPL-2.0', ('mozilla public license',)),
    ('permissive', 'MIT', ('permission is hereby granted, free of charge',)),
    ('permissive', 'BSD',
     ('redistribution and use in source and binary forms',)),
    ('permissive', 'ISC',
     ('permission to use, copy, modify, and/or distribute',)),
    ('copyleft', 'AGPL', ('gnu affero general public license',)),
    ('copyleft', 'LGPL', ('gnu lesser general public license',)),
    ('copyleft', 'GPL', ('gnu general public license',)),
)
# Categories a pack may be built from AND redistributed.
_REDISTRIBUTABLE_CATEGORIES = frozenset(
    {'permissive', 'weak-copyleft', 'copyleft'})
_LICENSE_FILE_STEMS = ('license', 'licence', 'copying')


def classify_license(text) -> tuple[str, str | None]:
    """Classify license ``text`` into ``(category, name)``.

    ``permissive`` / ``weak-copyleft`` / ``copyleft`` are redistribution-safe;
    ``source-available`` is not; ``unknown`` (unrecognized or empty) fails
    CLOSED — treated as not-safe by the gate. A non-free marker always wins over
    an open one (a Commons-Clause/BUSL rider makes an otherwise open grant
    non-free).
    """
    low = (text or '').lower()
    if not low.strip():
        return ('unknown', None)
    for name, markers in _NONFREE_LICENSE_MARKERS:
        if any(mk in low for mk in markers):
            return ('source-available', name)
    for category, name, markers in _OPEN_LICENSE_MARKERS:
        if any(mk in low for mk in markers):
            return (category, name)
    return ('unknown', None)


def detect_corpus_license(clone_dir) -> tuple[str, str | None]:
    """Classify a corpus clone's top-level license file. A directory with no
    recognizable license file → ``('unknown', None)`` (fail-closed)."""
    clone_dir = Path(clone_dir)
    if not clone_dir.is_dir():
        return ('unknown', None)
    for entry in sorted(clone_dir.iterdir()):
        low = entry.name.lower()
        if entry.is_file() and any(
                low == s or low.startswith(s + '.') or low.startswith(s + '-')
                for s in _LICENSE_FILE_STEMS):
            try:
                text = entry.read_text(encoding='utf-8', errors='replace')
            except OSError:
                return ('unknown', None)
            return classify_license(text)
    return ('unknown', None)


def nonfree_corpora(dest_dir) -> list:
    """Scan corpus clones under ``dest_dir`` (each marked with the fetch's
    ``.ariadne-corpus-sha``) and return ``[(repo, category, name)]`` for those
    NOT redistribution-safe under an open-source license. Empty when every
    corpus is open source — or when none were fetched (the mocked-build case,
    where there is simply nothing on disk to classify)."""
    from spool_acquire import _CORPUS_SHA_MARKER
    dest_dir = Path(dest_dir)
    bad = []
    for marker in sorted(dest_dir.glob(f'*/{_CORPUS_SHA_MARKER}')):
        repo_dir = marker.parent
        category, name = detect_corpus_license(repo_dir)
        if category not in _REDISTRIBUTABLE_CATEGORIES:
            bad.append((repo_dir.name, category, name))
    return bad


@dataclass(frozen=True)
class SpoolManifest:
    """A Spool pack's manifest (§9): identity, edition pin, certification.

    ``target_runtime`` is the runtime edition the pack was built for
    (§18.2); ``certified_docs`` is the publisher-certified official doc
    set (§18.3 tier 2).
    """
    environment: str | None = None
    version: str | None = None
    target_runtime: str | None = None
    certified_docs: tuple = ()
    checksum: str | None = None
    pack_format: int = 1
    # CRIT-10: the embedding identity the pack's vectors were produced under.
    # Install verifies these against the consumer's embedding config — a
    # pack's embeddings are meaningless (or crash the matrix) under a
    # different model/dimension. None when the pack carries no embeddings.
    embedding_model: str | None = None
    embedding_dim: int | None = None
    # D5 delta substrate: the per-repo commit shas this pack was built from,
    # so a later `create` can reuse unchanged repos instead of rebuilding.
    corpus_shas: dict = field(default_factory=dict)
    # Slice 2: the runtime pin's component-version map (corpus source →
    # component version, recipe-authored from the runtime's release notes) —
    # the right-hand side of the version_facts availability join, so
    # "is X available on MY runtime" resolves from the pin.
    runtime_components: dict = field(default_factory=dict)
    # Slice 3: the recipe's interaction-surface vocabularies (surface name →
    # keyword stems). The pack's tags are built from these at build time and
    # the CONSUMER matches questions against the same vocabularies — the
    # surface tier's two halves stay in lockstep.
    surfaces: dict = field(default_factory=dict)
    # Multi-word product name forms ('delta lake') that are not derivable
    # from component/corpus keys — the router's route-don't-admit name set.
    name_aliases: list = field(default_factory=list)
    # The aisle's advisory lens (designs/spool-expert-aisles.md §2): the
    # concern/opportunity/gotcha dimensions this expert applies to a caller's
    # code (databricks: parallelism · serialization · autolog-patching · …).
    # Consumed by the consult path to focus the aisle's answers.
    taxonomy: tuple = field(default_factory=tuple)
    # Extraction-coverage version the pack was built under (0 = pre-tracking →
    # treated as behind). Lets a consumer flag a pack whose SCIP intelligence
    # predates a coverage change; the fix is a rebuild (`ariadne spools
    # create`), since the corpus isn't shipped for a local re-index. See
    # docgen.extraction_coverage.
    extraction_coverage_version: int = 0
    # Upstream attribution (§18.1): per corpus repo, the LICENSE/NOTICE files
    # bundled under ``licenses/<repo>/`` in the pack, each with a sha256 so
    # install can integrity-check them. A tuple of
    # ``{repo, sha, files: ({name, sha256}, ...)}`` records. Empty for a pack
    # built from no fetched corpus (e.g. a hand-built test pack).
    attribution: tuple = field(default_factory=tuple)

    def to_dict(self) -> dict:
        """The manifest as a plain dict for YAML serialization — the inverse of
        :meth:`from_dict`. Field order matches the dataclass so the dumped
        manifest stays stable; tuples become lists so a SafeDumper accepts them.
        """
        return {
            'environment': self.environment,
            'version': self.version,
            'target_runtime': self.target_runtime,
            'certified_docs': list(self.certified_docs),
            'checksum': self.checksum,
            'pack_format': self.pack_format,
            'embedding_model': self.embedding_model,
            'embedding_dim': self.embedding_dim,
            'corpus_shas': dict(self.corpus_shas),
            'runtime_components': dict(self.runtime_components),
            'surfaces': {k: list(v) for k, v in self.surfaces.items()},
            'name_aliases': list(self.name_aliases),
            'taxonomy': list(self.taxonomy),
            'extraction_coverage_version': self.extraction_coverage_version,
            'attribution': [
                {'repo': r['repo'], 'sha': r['sha'],
                 'files': [dict(f) for f in r['files']],
                 **({'license_name': r['license_name']}
                    if r.get('license_name') else {})}
                for r in self.attribution
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SpoolManifest':
        missing = [f for f in _MANIFEST_REQUIRED_FIELDS if not data.get(f)]
        if missing:
            raise SpoolError(
                f'spool manifest missing required field(s): '
                f'{", ".join(missing)}',
            )
        environment = str(data['environment'])
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', environment):
            # HIGH-1: environment is attacker-controlled and uncovered by the
            # checksum; it becomes both a filesystem path (Path(cache)/env) and
            # the spool:<env> source id. Require a bare identifier so it can
            # neither traverse the cache dir nor forge a source namespace.
            raise SpoolError(
                f'spool manifest environment {environment!r} is not a bare '
                f'identifier (no path separators, .., or absolute paths)',
            )
        emb_dim = data.get('embedding_dim')
        try:
            return cls(
                environment=environment,
                version=str(data['version']),
                target_runtime=data['target_runtime'],
                certified_docs=tuple(data.get('certified_docs') or ()),
                checksum=data['checksum'],
                pack_format=int(data.get('pack_format', 1)),
                embedding_model=data.get('embedding_model'),
                embedding_dim=int(emb_dim) if emb_dim is not None else None,
                corpus_shas=dict(data.get('corpus_shas') or {}),
                runtime_components=dict(
                    data.get('runtime_components') or {}),
                surfaces={
                    str(k): [str(s) for s in (v or [])]
                    for k, v in (data.get('surfaces') or {}).items()},
                name_aliases=[
                    str(a) for a in (data.get('name_aliases') or [])],
                taxonomy=tuple(data.get('taxonomy') or ()),
                extraction_coverage_version=int(
                    data.get('extraction_coverage_version', 0) or 0),
                attribution=tuple(
                    {'repo': str(r.get('repo', '')),
                     'sha': str(r.get('sha', '')),
                     'files': tuple(
                         {'name': str(f.get('name', '')),
                          'sha256': str(f.get('sha256', ''))}
                         for f in (r.get('files') or ())),
                     **({'license_name': str(r['license_name'])}
                        if r.get('license_name') else {})}
                    for r in (data.get('attribution') or ())),
            )
        except (ValueError, TypeError) as exc:
            # CRIT-3c: a type-invalid field (non-numeric pack_format/
            # embedding_dim, non-iterable certified_docs) must fail CLOSED as a
            # SpoolError — resolve_spools guards on SpoolError only, so a raw
            # ValueError/TypeError here would escape and crash the query path.
            raise SpoolError(
                f'spool manifest has a malformed field: {exc}',
            ) from exc

    @classmethod
    def load(cls, path) -> 'SpoolManifest':
        return cls.from_dict(load_yaml_mapping(path, SpoolError))


@dataclass(frozen=True)
class SpoolSetting:
    """Per-spool settings from the ``spools:`` mapping (§18.6.4).

    ``runtime`` is the project's declared target runtime for this Spool
    (the pin the manifest is checked against); ``None`` = no pin declared.
    ``projects`` is the list of the user's sources this spool cross-checks
    (its cross-source themes); empty = no cross-check yet.
    """
    runtime: str | None = None
    projects: tuple[str, ...] = ()


@dataclass(frozen=True)
class SpoolGap:
    """A structured honest-gap outcome (§3, §18.6.4): the spool did not
    load, and this says exactly why — never a silent skip."""
    spool: str
    reason: str  # 'missing-pack' | 'corrupt-pack' | 'runtime-unpinned' | 'runtime-mismatch'
    message: str


@dataclass(frozen=True)
class SpoolRegistration:
    """An enabled, pin-verified spool ready for scope inclusion (§17)."""
    spool: str
    manifest: SpoolManifest
    kind: str = 'spool'


@dataclass(frozen=True)
class SpoolResolution:
    """The outcome of resolving the ``spools:`` mapping at load time."""
    registered: dict
    gaps: tuple

    def scope_sources(self) -> frozenset[str]:
        """The reserved spool source ids eligible for query-scope inclusion.

        Namespaced (``spool:<name>``, CRIT-9) so a spool never shares a
        source namespace with real user code. The scope union + the
        origin-axis gate + the ask fence all key on these ids.
        """
        return frozenset(spool_source_id(name) for name in self.registered)

    def fingerprint(self) -> str:
        """A stable fingerprint of this registered set (CRIT-11), folded into
        result-cache keys so enable/disable/update self-invalidate. Empty when
        nothing is registered (non-spool projects keep their old cache keys).
        Keyed on (name, manifest checksum) — appears/drops/changes on each of
        the three transitions."""
        import hashlib

        if not self.registered:
            return ''
        parts = sorted(
            f'{name}@{reg.manifest.checksum or ""}'
            for name, reg in self.registered.items()
        )
        return hashlib.sha256(';'.join(parts).encode()).hexdigest()[:16]


# Provisional gate constants (§18.6.3 — calibrate at slice c2 with real
# score distributions; parameters everywhere so calibration is a re-tune,
# not a rewrite).
_GATE_MIN_STRONG_HITS = 3
_GATE_RELATIVE_THRESHOLD = 0.85


def partition_tier2(lite_docs, spool_sources=frozenset()):
    """Split candidates into (tier-1, tier-2) — §18.6.3 gate mechanism.

    Tier-2 gates on BOTH axes (§18.6.2, HIGH-1): a doc is tier-2 only when
    it is ``provenance == 'official'`` AND its source is an active spool.
    So a user's own doc tagged 'official' is never suppressed, and with no
    spool active nothing is tier-2. Everything else — code, human-doc,
    stale, user-official — is tier-1 and competes normally.
    """
    spool_sources = frozenset(spool_sources)
    tier1, tier2 = [], []
    for doc in lite_docs:
        is_official = doc.metadata.get('provenance') == OFFICIAL_DOC_PROVENANCE
        from_spool = getattr(doc, 'source_name', None) in spool_sources
        if is_official and from_spool:
            tier2.append(doc)
        else:
            tier1.append(doc)
    return tier1, tier2


def scarcity_gate_open(
    ranked,
    *,
    min_strong_hits=_GATE_MIN_STRONG_HITS,
    relative_threshold=_GATE_RELATIVE_THRESHOLD,
) -> bool:
    """§18.6.3 predicate (A): open iff the tier-1 layer is scarce.

    Scarce = fewer than ``min_strong_hits`` results scoring at least
    ``relative_threshold × top_score``. The relative form is robust to
    embedding-scale differences. Empty or zero-top results are scarce by
    definition.
    """
    if not ranked:
        return True
    top_score = ranked[0][1]
    if not top_score or top_score <= 0:
        return True
    strong = sum(
        1 for _, score in ranked
        if score is not None and score >= relative_threshold * top_score
    )
    return strong < min_strong_hits


@dataclass(frozen=True)
class GatedRank:
    """The gated ranking outcome (§18.6.3): results + what the gate did."""
    ranked: list
    gate_opened: bool
    tier2_ranked: bool


async def rank_with_scarcity_gate(
    rank,
    tier1_ids,
    tier2_ids,
    *,
    min_strong_hits=_GATE_MIN_STRONG_HITS,
    relative_threshold=_GATE_RELATIVE_THRESHOLD,
) -> GatedRank:
    """Rank tier-1; re-rank with tier-2 included iff the gate opens.

    ``rank`` is an async callable(ids) -> [(doc_id, score), ...]. When the
    gate stays closed (code answered strongly) tier-2 is never ranked —
    fill, don't dilute (§18.6.3). The gate state travels with the result
    so callers can surface the honest gap (gate open, nothing to fill).
    """
    tier1 = list(tier1_ids)
    ranked = await rank(tier1)
    gate_opened = scarcity_gate_open(
        ranked,
        min_strong_hits=min_strong_hits,
        relative_threshold=relative_threshold,
    )
    tier2_ranked = bool(tier2_ids) and gate_opened
    if tier2_ranked:
        ranked = await rank(tier1 + list(tier2_ids))
    return GatedRank(
        ranked=ranked, gate_opened=gate_opened, tier2_ranked=tier2_ranked,
    )


def spool_gap_hint(*, gate_opened: bool, tier2_present: bool,
                   spools_registered: bool) -> str | None:
    """The §18.6.1 live-consult trigger: the code layer ran thin AND no
    certified tier-2 docs existed to fill AND a spool is active. Returns
    the honest-gap message, or None (never noise for non-spool projects)."""
    if gate_opened and not tier2_present and spools_registered:
        return (
            'The code layer ran thin for this query and the installed '
            'spool packs no certified official docs for it — consider '
            "consulting the environment's official documentation "
            '(live-consult) and citing the page.'
        )
    return None


def default_cache_dir(config) -> Path:
    """The default pack cache location: ``<config dir>/.ariadne/spools``."""
    return config.config_dir / '.ariadne' / 'spools'


def spool_scope_fingerprint(config, *, cache_dir=None) -> str:
    """The registered-spool fingerprint (CRIT-11) — thin wrapper resolving
    once and delegating to ``SpoolResolution.fingerprint``. Prefer resolving
    the ``SpoolResolution`` once per request and reading both ``fingerprint``
    and ``scope_sources`` off it (see ``active_spool_sources``)."""
    return resolve_spools(config, cache_dir=cache_dir).fingerprint()


def active_spool_sources(config, *, cache_dir=None) -> 'frozenset[str]':
    """The reserved source ids of the registered spools — the one-call form
    of the ``resolve_spools(...).scope_sources()`` idiom (CRIT-9 scope)."""
    return resolve_spools(config, cache_dir=cache_dir).scope_sources()


def resolve_spools(config, *, cache_dir=None) -> SpoolResolution:
    """Resolve enabled spools against the pack cache (§18.2, §18.6.4).

    ``cache_dir`` defaults to ``default_cache_dir(config)``. Fail-closed on an
    unpinned or mismatched runtime: an enable MUST pin the pack's runtime, else
    the spool is refused with a gap and nothing registers.
    """
    if cache_dir is None:
        cache_dir = default_cache_dir(config)
    registered: dict[str, SpoolRegistration] = {}
    gaps: list[SpoolGap] = []
    for name, setting in enabled_spools(config).items():
        manifest_path = Path(cache_dir) / name / 'manifest.yaml'
        if not manifest_path.exists():
            gaps.append(SpoolGap(
                spool=name,
                reason='missing-pack',
                message=(
                    f"spool '{name}' is enabled but no pack is cached at "
                    f'{manifest_path.parent} — fetch it first'
                ),
            ))
            continue
        try:
            manifest = SpoolManifest.load(manifest_path)
        except SpoolError as exc:
            # CRIT-3: a damaged cache must degrade to a visible gap, never
            # crash the query path that resolution runs on.
            gaps.append(SpoolGap(
                spool=name,
                reason='corrupt-pack',
                message=(
                    f"spool '{name}' has an unreadable/invalid cached "
                    f'manifest ({exc}) — reinstall the pack'
                ),
            ))
            continue
        if setting.runtime is None:
            # H1: an unpinned enable would accept ANY signed version of this
            # environment (a substitution/downgrade vector once packs are
            # signed). Fail closed — require the pin — and name the cached
            # pack's runtime so the fix is obvious.
            gaps.append(SpoolGap(
                spool=name,
                reason='runtime-unpinned',
                message=(
                    f"spool '{name}' refused: no runtime pin. An unpinned "
                    f'spool would accept ANY signed version (a substitution/'
                    f'downgrade vector), so set spools.{name}.runtime to the '
                    f'cached pack runtime {manifest.target_runtime!r} in '
                    f'ariadne.yaml'
                ),
            ))
            continue
        if setting.runtime != manifest.target_runtime:
            gaps.append(SpoolGap(
                spool=name,
                reason='runtime-mismatch',
                message=(
                    f"spool '{name}' refused: the project targets runtime "
                    f'{setting.runtime!r} but the cached pack was built '
                    f'for {manifest.target_runtime!r}'
                ),
            ))
            continue
        registered[name] = SpoolRegistration(spool=name, manifest=manifest)
    return SpoolResolution(registered=registered, gaps=tuple(gaps))


def enabled_spools(config) -> dict[str, SpoolSetting]:
    """The enabled-spools view of a ``Config`` (§18.6.4): name → setting.

    ``true`` enables a spool with no runtime pin; ``false`` (and an absent
    ``spools:`` key) contributes nothing.
    """
    raw = config.to_dict().get('spools') or {}
    if not isinstance(raw, dict):
        raise SpoolError(
            f"'spools' must be a mapping of spool name to setting, "
            f'got {type(raw).__name__}',
        )
    enabled: dict[str, SpoolSetting] = {}
    for name, value in raw.items():
        if value is True:
            enabled[name] = SpoolSetting()
        elif value is False or value is None:
            continue
        elif isinstance(value, dict):
            if not value.get('enabled', True):
                continue
            enabled[name] = SpoolSetting(
                runtime=value.get('runtime'),
                projects=tuple(value.get('projects') or ()),
            )
        else:
            raise SpoolError(
                f"spools.{name}: expected true/false or a settings "
                f'mapping, got {value!r}',
            )
    return enabled


@dataclass(frozen=True)
class SpoolReconcile:
    """Outcome of reconciling a spool's theme partitions to its config.

    added: projects newly clustered against the spool this run.
    removed: association keys whose themes were deleted (dropped projects).
    dirty_theme_count: spool themes now needing (paid) summarization — the
        figure the caller shows and gates on before running the summarizer.
    skipped: projects whose association was unchanged since its last reconcile,
        so the (costly) edge rebuild + re-cluster was skipped (HIGH-A).
    """
    added: tuple
    removed: tuple
    dirty_theme_count: int
    skipped: tuple = ()


def _association_content_hash(library, scope) -> str:
    """Content signature of an association's {project, spool} catalog docs —
    their ids + per-element sync hashes. Reconcile compares this to the stored
    signature to skip rebuilding an unchanged association (HIGH-A)."""
    import hashlib

    from docgen.graph_builder import _scoped_view
    scoped = _scoped_view(library, None, scope=scope)
    parts = sorted(
        f'{d.id}\x00{(d.metadata or {}).get("sha_at_sync", "")}'
        for d in scoped.list_documents_lite(content_type='catalog')
    )
    return hashlib.sha256('\n'.join(parts).encode('utf-8')).hexdigest()


def reconcile_spool_themes(
    library, config, spool_name, *,
    min_cluster_size: int = 3, k: int = 5, min_sim: float = 0.6,
) -> SpoolReconcile:
    """Make a spool's theme partitions match ``spools.<name>.projects``.

    Idempotent. For each opted-in project it (re)builds cross-source semantic
    edges and clusters the {project, spool} pass — free, deterministic work
    that leaves the new themes dirty. For each project no longer opted in it
    deletes that partition's themes (also free). The paid step — summarizing
    the dirty themes — is left to the caller, which gates it on the reported
    count. Clustering runs below the closure chokepoint, like the base build.
    """
    from docgen.cluster import _association_key, cluster_themes
    from docgen.graph_builder import build_semantic_edges

    setting = enabled_spools(config).get(spool_name)
    if setting is None:
        raise SpoolError(f'spool {spool_name!r} is not enabled in ariadne.yaml')
    spool_source = spool_source_id(spool_name)
    desired = {
        _association_key(frozenset({project, spool_source})): project
        for project in setting.projects
    }
    current = {
        assoc for assoc in library.distinct_theme_associations()
        if spool_source in assoc.split('|')
    }

    removed = tuple(sorted(current - set(desired)))
    for assoc in removed:
        library.delete_themes_for_association(assoc)

    # Re-cluster each desired association whose {project, spool} content CHANGED
    # since its last reconcile; skip the rest (HIGH-A). Rebuilding is what keeps
    # cross-source themes tracking base-project changes (the edges are transient
    # — base theme maintenance rebuilds over configured sources only, dropping
    # the cross-source ones), but rebuilding an UNCHANGED association would
    # re-index the whole spool corpus for nothing, so enabling/refreshing one
    # project must not re-index every other. The per-scope rebuild is preserved
    # exactly — only redundant rebuilds are skipped, never merged into a shared
    # index (which would change the derived edges).
    added = []
    skipped = []
    for assoc, project in desired.items():
        scope = frozenset({project, spool_source})
        content_hash = _association_content_hash(library, scope)
        if (assoc in current
                and library.get_spool_assoc_hash(assoc) == content_hash):
            skipped.append(project)
            continue
        build_semantic_edges(library, scope=scope, k=k, min_sim=min_sim)
        cluster_themes(library, scope=scope, min_cluster_size=min_cluster_size)
        library.set_spool_assoc_hash(assoc, content_hash)
        if assoc not in current:
            added.append(project)

    dirty = sum(
        1 for assoc in desired
        for theme in library.list_themes(coherent_only=False, association=assoc)
        if theme.dirty
    )
    return SpoolReconcile(
        added=tuple(added), removed=removed, dirty_theme_count=dirty,
        skipped=tuple(skipped),
    )


def build_spool_internal_themes(
    library, spool_name, corpus_sources, *,
    min_cluster_size: int = 3, k: int = 5, min_sim: float = 0.6,
) -> int:
    """Cluster a spool's OWN corpus into spool-internal themes.

    The single-source analog of :func:`reconcile_spool_themes`: it (re)builds
    semantic edges over the corpus scope and clusters it in association-only
    mode. ``cross_source_only=False`` keeps the corpus's internal clusters (a
    cross-source pass would drop them), and they are tagged under the spool's
    reserved ``spool:<name>`` association so they gate by enable and never
    occupy the base '' pass (that base-pass tagging is the leak). Clustering
    is free/deterministic and leaves the new themes dirty; the caller
    summarizes the returned count.
    """
    from docgen.cluster import cluster_themes
    from docgen.graph_builder import build_semantic_edges

    spool_source = spool_source_id(spool_name)
    scope = frozenset(corpus_sources)
    build_semantic_edges(library, scope=scope, k=k, min_sim=min_sim)
    cluster_themes(
        library, scope=scope, association=spool_source,
        cross_source_only=False, min_cluster_size=min_cluster_size,
    )
    return sum(
        1
        for theme in library.list_themes(
            coherent_only=False, association=spool_source,
        )
        if theme.dirty
    )


def delta_corpus_plan(prior_shas: dict, new_shas: dict) -> tuple:
    """Split a new corpus into (reuse, rebuild) against a prior build.

    A repo whose pinned sha matches the prior pack's is reused — its docs
    carry over unchanged. A repo with a changed sha, or one absent from the
    prior pack, is rebuilt. Returns ``(reuse, rebuild)`` as sorted name
    tuples; this is the free, deterministic decision that lets a version bump
    (e.g. dbr17.3 → 18.0) re-index only what actually moved.
    """
    reuse = tuple(sorted(
        name for name, sha in new_shas.items()
        if prior_shas.get(name) == sha
    ))
    rebuild = tuple(sorted(set(new_shas) - set(reuse)))
    return reuse, rebuild
