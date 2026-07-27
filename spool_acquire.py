"""Builder acquisition with consent-before-fetch + unified ``spools create``.

A packfile declares the corpus a Spool build fetches: repo URL + tag +
**SHA** per entry (tags are mutable; the SHA is the pin). Before ANY
acquisition the consent prompt shows exactly what will be fetched, and
``--approve`` skips it for CI. Clones are shallow (``--depth 1`` at the
tag), pinned to the exact SHA, then stripped of ``.git`` and marked with
``.ariadne-corpus-sha`` so a re-run reuses/adopts/refetches its own
workspace instead of refusing. Runs on the BUILDER's machine only.

``setup_recipe`` drives the interactive recipe build: the repo SET comes
from a built-in recipe (or GitHub discovery by name); the user picks each
version from the COMPATIBLE tags, resolved in a discovered dependency
order as a cascade (an upstream pick constrains the groups after it).

Design: IMPLEMENT.md (b3) · designs/spool-environment-plugin.md §18.
"""
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from spools import (SpoolError, is_scip_eligible, load_yaml_mapping,
                    nonfree_corpora, unsupported_corpus_language)


@dataclass(frozen=True)
class CorpusEntry:
    url: str
    sha: str
    tag: str | None = None


@dataclass(frozen=True)
class Packfile:
    runtime: str
    corpus: dict
    certify: tuple = ()


@dataclass(frozen=True)
class AcquireResult:
    accepted: bool
    cloned: tuple = ()


@dataclass(frozen=True)
class CreateResult:
    accepted: bool
    pack_path: str | None = None


_RECIPES_DIR = Path(__file__).parent / 'spool_content' / 'recipes'

# A spool build documents the library SURFACE, not the test suite: test dirs
# are pruned from the walk and Scala/Java test-file globs excluded (Python test
# globs are already a global default).
_SPOOL_IGNORE_TEST_DIRS = ('test', 'tests', 'src/test', 'it')
_SPOOL_IGNORE_TEST_GLOBS = (
    '**/*Suite.scala', '**/*Spec.scala', '**/*Test.java', '**/*Tests.java',
)
# A spool build leaves the heavy doc types off by default (opt-in in the
# picker) to cut cost — a reference pack wants explanation + gotcha + catalog.
_SPOOL_DOC_TYPES_OFF = ('architecture', 'qa', 'diagram')

# Written into each acquired corpus so a re-run recognises its OWN workspace
# (reuse / adopt / refetch) rather than refusing a leftover.
_CORPUS_SHA_MARKER = '.ariadne-corpus-sha'


def load_packfile(path) -> Packfile:
    """Load + validate a packfile; loud on any missing pin."""
    data = load_yaml_mapping(path, SpoolError)
    runtime = data.get('runtime')
    corpus_raw = data.get('corpus') or {}
    if not runtime or not corpus_raw:
        raise SpoolError(
            f'packfile {path} must declare `runtime` and a non-empty '
            f'`corpus`',
        )
    corpus = {}
    for name, entry in corpus_raw.items():
        entry = entry or {}
        missing = [k for k in ('url', 'sha') if not entry.get(k)]
        if missing:
            raise SpoolError(
                f"packfile corpus entry '{name}' missing "
                f"{', '.join(missing)} — the sha IS the pin (tags are "
                f'mutable), it cannot be omitted',
            )
        corpus[name] = CorpusEntry(
            url=str(entry['url']),
            sha=str(entry['sha']),
            tag=entry.get('tag'),
        )
    return Packfile(
        runtime=str(runtime),
        corpus=corpus,
        certify=tuple(data.get('certify') or ()),
    )


def _ask(prompt, label: str, default: str) -> str:
    """Prompt with a default shown in ``[...]``; an empty answer keeps it."""
    suffix = f' [{default}]' if default else ''
    resp = prompt(f'{label}{suffix}: ').strip()
    return resp or default


# --------------------------------------------------------------------------
# Version-cascade helpers. Each takes an injectable seam (``ask`` / ``tags_fn``
# / ``search_fn``) so tests drive them offline; the operational defaults
# (`_llm_ask`, `_github_search`) reach the network and fail SOFT.
# --------------------------------------------------------------------------

def _default_llm_ask(prompt: str) -> str:
    """One-shot LLM text query via the configured provider (``llm.chat_complete``).

    The default seam for the compatibility/order helpers; raises ``ValueError``
    if the provider API key is missing (callers warn + fail soft). Tests inject
    ``ask=`` and mock ``llm.chat_complete`` at the network hop.
    """
    import asyncio

    import llm
    return asyncio.run(llm.chat_complete([{'role': 'user', 'content': prompt}]))


def _github_slug(url):
    """``owner/repo`` for a github.com URL, else None (non-github → no API)."""
    match = re.match(
        r'^https?://github\.com/([^/]+/[^/]+?)(?:\.git)?/?$', str(url))
    return match.group(1) if match else None


def _github_tree(slug, sha) -> dict:
    """GitHub recursive tree for ``slug`` at ``sha`` (files-count source)."""
    import httpx
    resp = httpx.get(
        f'https://api.github.com/repos/{slug}/git/trees/{sha}',
        params={'recursive': '1'}, timeout=15,
    )
    return resp.json() if resp.status_code == 200 else {}


def _remote_file_count(url, sha, *, fetch=None) -> int:
    """Blob (file) count in ``url`` at ``sha`` — the progress-bar total.

    GitHub-only (via the trees API); non-github → 0 with NO fetch. A truncated
    tree can't be trusted for a determinate total → 0; any API failure fails
    soft to 0 (the fetch still proceeds, just without a bar total).
    """
    slug = _github_slug(url)
    if slug is None:
        return 0
    fetch = fetch or _github_tree
    try:
        tree = fetch(slug, sha)
    except Exception:
        return 0
    if not isinstance(tree, dict) or tree.get('truncated'):
        return 0
    return sum(1 for entry in tree.get('tree', []) if entry.get('type') == 'blob')


def _count_files(path) -> int:
    """Recursive file count under ``path`` (0 if it does not exist)."""
    root = Path(path)
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob('*') if p.is_file())


def _github_search(name: str) -> list:
    """Raw GitHub repo-search items for ``name`` (the default ``search_fn``
    seam for ``_search_repos``). Fails SOFT to []."""
    try:
        import httpx
        resp = httpx.get(
            'https://api.github.com/search/repositories',
            params={'q': f'{name} in:name', 'sort': 'stars', 'order': 'desc'},
            timeout=10,
        )
        return resp.json().get('items', []) if resp.status_code == 200 else []
    except Exception:
        return []


def _search_repos(name, *, search_fn=None) -> list:
    """EXACT-name (case-insensitive) repo matches, most-starred first, mapped to
    ``{full_name, url, stars, language, name}``. Real search results only (never
    a guessed URL); ``search_fn`` is the injectable seam. Fails SOFT to []."""
    search_fn = search_fn or _github_search
    try:
        items = search_fn(name) or []
    except Exception:
        return []
    exact = [i for i in items
             if str(i.get('name', '')).lower() == str(name).lower()]
    exact.sort(key=lambda i: i.get('stargazers_count', 0), reverse=True)
    return [{
        'full_name': i.get('full_name'),
        'url': i.get('clone_url'),
        'stars': i.get('stargazers_count', 0),
        'language': i.get('language'),
        'name': i.get('name'),
    } for i in exact]


def _repo_tags(url: str) -> list:
    """A repo's release tags, newest-first, pre-releases filtered out.

    Parses ``git ls-remote --tags`` (peeled ``^{}`` deduped); degrades to []
    when git fails (offline / bad url → the picker falls back to the pin).
    """
    try:
        out = _git('.', 'ls-remote', '--tags', str(url))
    except SpoolError:
        return []
    tags = set()
    for line in out.splitlines():
        if '\t' not in line:
            continue
        ref = line.split('\t', 1)[1].strip()
        if not ref.startswith('refs/tags/'):
            continue
        tag = ref[len('refs/tags/'):]
        if tag.endswith('^{}'):
            tag = tag[:-3]
        if re.search(r'-[A-Za-z]', tag):     # rc / preview / alpha / beta …
            continue
        tags.add(tag)

    def _ver_key(t):
        return [int(x) for x in re.findall(r'\d+', t)] or [0]

    return sorted(tags, key=_ver_key, reverse=True)


def _extract_json(reply: str):
    """First JSON object/array in ``reply``; None if none / unparseable."""
    for pattern in (r'\{.*\}', r'\[.*\]'):
        match = re.search(pattern, reply, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except (ValueError, TypeError):
                return None
    return None


def _query_compat(environment, runtime, repos, chosen=None, *, ask=None) -> dict:
    """Each repo's COMPATIBLE version line given the runtime AND the versions
    already chosen (the cascade condition). A real-time LLM query, parsed from
    the model's JSON; fails SOFT to {} (→ pin + warn) on any bad shape."""
    if not repos:
        return {}
    chosen = chosen or {}
    ask = ask or _default_llm_ask
    prompt = (
        f'For the {environment!r} runtime {runtime!r}, which release LINE of '
        f'each of these repos is compatible: {list(repos)}? '
        f'Versions already chosen (constrain the rest): {dict(chosen)}. '
        f'Reply ONLY a JSON object {{repo: "major.minor" or null}}.'
    )
    try:
        obj = _extract_json(ask(prompt))
    except Exception as exc:
        print(f'  warning: version-compatibility query failed ({exc}) — '
              f'falling back to the blessed pins')
        return {}
    if not isinstance(obj, dict):
        return {}
    return {r: str(obj[r]) for r in repos if obj.get(r)}


def _discover_order(environment, runtime, repos, ask=None) -> list:
    """Dependency order (root first) for the version cascade — a JSON array
    from the model, validated to be a permutation of ``repos``. Fails SOFT to
    the given order (never a bad root); a single repo needs no query."""
    repos = list(repos)
    if len(repos) <= 1:
        return repos
    ask = ask or _default_llm_ask
    prompt = (
        f'Order these {environment!r} repos so each depends only on earlier '
        f'ones (root first): {repos}. Reply ONLY a JSON array of the names.'
    )
    try:
        arr = _extract_json(ask(prompt))
    except Exception:
        return repos
    if isinstance(arr, list) and sorted(map(str, arr)) == sorted(repos):
        return [str(x) for x in arr]
    return repos


def _tag_matches_line(tag, line) -> bool:
    """Does ``tag`` belong to version ``line`` (e.g. 'v4.0.1' in '4.0')?"""
    if not line:
        return False
    normalized = str(tag).lstrip('vV')
    line = str(line)
    return normalized == line or normalized.startswith(line + '.')


def _is_moving_ref(ref) -> bool:
    """A branch-like placeholder (never a reproducible spool pin)."""
    return str(ref).lower() in ('main', 'master', 'head', 'trunk', 'develop', 'dev')


def _select_cascade(prompt, order, candidates_fn) -> dict:
    """Resolve a version per repo in dependency order, feeding each pick
    forward. ``candidates_fn(repo, chosen)`` returns ``(options, default,
    warn)``. A single confident compatible option (``warn=False``) is
    auto-chosen and merely NOTIFIED; anything else prompts (numbered pick /
    typed name / Enter=default) — including a lone pin under ``warn`` so the
    escape hatch survives."""
    chosen = {}
    for repo in order:
        options, default, warn = candidates_fn(repo, chosen)
        if len(options) == 1 and not warn:
            chosen[repo] = options[0]
            print(f'  {repo}: {options[0]} (only compatible version)')
            continue
        if warn:
            print(f'  {repo}: could not resolve a compatible line — offering '
                  f'the pin {default!r} only (type a tag to override)')
        listing = '  '.join(f'{i + 1}={t}' for i, t in enumerate(options))
        resp = prompt(f'{repo} tag (# or name) [{default}]  {listing}: ').strip()
        if not resp:
            chosen[repo] = default
        elif resp.isdigit() and 1 <= int(resp) <= len(options):
            chosen[repo] = options[int(resp) - 1]
        else:
            chosen[repo] = resp
    return chosen


def _discover_environment(environment, prompt, search_fn=None) -> dict:
    """Build a recipe base for an environment with no shipped recipe by
    searching GitHub for a repo NAMED ``environment`` (a single-repo env: its
    chosen version becomes the runtime). One match is used; several are shown to
    pick by number (or paste a URL to override); none prompts for a URL, and a
    blank answer aborts loud. The repo's primary language becomes the declared
    spool language (the grounding gate checks it)."""
    repos = _search_repos(environment, search_fn=search_fn)
    if len(repos) == 1:
        repo = repos[0]
    elif len(repos) > 1:
        for i, r in enumerate(repos):
            print(f'  {i + 1}. {r["full_name"]}  ({r["stars"]}★)  {r["url"]}')
        resp = prompt(
            f'Which {environment} repo? [# or paste a URL, Enter=1]: ').strip()
        if resp.startswith(('http://', 'https://', 'git@', 'ssh://')):
            repo = {'url': resp, 'language': None}
        elif resp.isdigit() and 1 <= int(resp) <= len(repos):
            repo = repos[int(resp) - 1]
        else:
            repo = repos[0]
    else:
        url = prompt(
            f'No GitHub repo found named {environment!r}; paste a repo URL '
            f'(blank to abort): '
        ).strip()
        if not url:
            raise SpoolError(
                f'no built-in recipe for {environment!r}, no GitHub match, and '
                f'no URL given — nothing to build',
            )
        repo = {'url': url, 'language': None}
    base = {
        'name': environment,
        'version': '1.0.0',
        'corpus': {environment: {'url': repo.get('url')}},
    }
    language = repo.get('language')
    if language:
        base['languages'] = [str(language).lower()]
    return base


def setup_recipe(environment=None, *, out_path='spools.yaml', prompt=None,
                 available=None, tags_fn=None, compat_fn=None,
                 order_fn=None, search_fn=None) -> Path:
    """Interactive recipe setup for the unified ``spools create``.

    The repo SET comes from the environment's built-in recipe. The RUNTIME is
    chosen first — it drives compatibility — then the dependency ORDER is
    discovered (which component the runtime governs, which are constrained by
    others) BEFORE any prompt, and versions are resolved as a CASCADE in that
    order: each repo's live git tags are narrowed to the compatible line
    (resolved by a real-time LLM query, so any spool works) given the runtime
    AND the versions already picked, so an upstream pick constrains the groups
    that follow. When no line resolves, the picker offers the blessed pin alone
    (plus a warning) — never the full tag list. An existing ``out_path`` seeds
    the defaults. Returns the path.
    """
    prompt = prompt or input
    tags_fn = tags_fn or _repo_tags
    compat_fn = compat_fn or _query_compat
    order_fn = order_fn or _discover_order
    out_path = Path(out_path)
    if available is None:
        available = sorted(p.stem for p in _RECIPES_DIR.glob('*.yaml'))
    if not available:
        raise SpoolError('no built-in spool recipes available')

    if environment is None:
        environment = prompt(
            f'Which spool to create? (built-in: {", ".join(available)}; or any '
            f'GitHub repo name — version numbers come later): '
        ).strip() or available[0]
    # A shipped recipe wins; otherwise discover the repo(s) on GitHub by name.
    recipe_file = _RECIPES_DIR / f'{environment}.yaml'
    if recipe_file.exists():
        base = yaml.safe_load(recipe_file.read_text(encoding='utf-8'))
    else:
        base = _discover_environment(environment, prompt, search_fn)

    # An existing spools.yaml seeds the defaults over the recipe/discovery.
    if out_path.exists():
        existing = yaml.safe_load(out_path.read_text(encoding='utf-8')) or {}
        if existing.get('corpus'):
            base = existing

    # Runtime first: it drives which component versions are compatible. A
    # single-repo environment (discovered repo, OR a recipe with no `runtime:`)
    # has no separate edition — its chosen corpus version IS the edition, pinned
    # after the cascade below. Skip the prompt there.
    version_is_runtime = not base.get('runtime')
    if not version_is_runtime:
        base['runtime'] = _ask(prompt, 'Runtime edition', base['runtime'])
    corpus = base.get('corpus', {})
    runtime = base.get('runtime', '')
    order = order_fn(environment, runtime, list(corpus)) or list(corpus)

    def _candidates(repo, chosen):
        """A repo's COMPATIBLE tags given the runtime and the versions picked
        so far: fetch the live tags, narrow to the compatible line, and fall
        back to the blessed pin ALONE when no line resolves — never the full
        tag list. A typed tag stays an escape hatch."""
        spec = corpus[repo]
        print(f'  fetching tags for {repo}…')
        all_tags = tags_fn(spec.get('url', ''))
        line = compat_fn(environment, runtime, [repo], chosen).get(repo)
        matched = ([t for t in all_tags if _tag_matches_line(t, line)]
                   if line else [])
        pin = spec.get('tag', '')
        if matched:
            return matched, (pin if pin in matched else matched[0]), False
        # No compatible line resolved. Offer the blessed pin — but only if it is
        # a real version; a moving-branch placeholder is not a valid pin, so
        # fall back to the live release tags (newest default), never the branch.
        if pin and not _is_moving_ref(pin):
            return [pin], pin, True
        if all_tags:
            return all_tags, all_tags[0], True
        return ([pin] if pin else []), pin, True

    for repo, tag in _select_cascade(prompt, order, _candidates).items():
        corpus[repo]['tag'] = tag

    # Discovered / runtime-less env: the chosen corpus version IS the edition.
    if not base.get('runtime') and order:
        base['runtime'] = corpus[order[0]].get('tag', '') or environment

    out_path.write_text(
        yaml.safe_dump(base, sort_keys=False), encoding='utf-8',
    )
    return out_path


def _load_spoolfile(path) -> tuple[str, str, dict]:
    """Load a spools.yaml (name + version + raw data); sha may be absent
    per entry — ``create`` resolves it from the tag before consent."""
    data = load_yaml_mapping(path, SpoolError)
    name = data.get('name')
    if not name or not data.get('runtime') or not data.get('corpus'):
        raise SpoolError(
            f'{path} must declare `name`, `runtime` and a non-empty `corpus`',
        )
    return str(name), str(data.get('version') or '1.0.0'), data


def _resolve_sha(url: str, tag: str) -> str:
    """Resolve a tag to its COMMIT sha.

    Annotated tags list twice: the tag object at ``refs/tags/T`` and the peeled
    commit at ``refs/tags/T^{}`` — ls-remote sorts the tag object first.
    Preference: peeled commit > lightweight tag > branch; never the tag object.
    """
    candidates = [f'refs/tags/{tag}^{{}}', f'refs/tags/{tag}', f'refs/heads/{tag}']
    out = _git('.', 'ls-remote', str(url), *candidates)
    refs = {}
    for line in out.splitlines():
        if line.strip():
            sha, ref = line.split('\t', 1)
            refs[ref.strip()] = sha.strip()
    for candidate in candidates:
        if candidate in refs:
            return refs[candidate]
    raise SpoolError(f'cannot resolve {tag!r} in {url} — no such tag/branch')


def create_spool(spoolfile_path, *, dest_dir, out_path, approve: bool,
                 confirm=input, phases: dict | None = None,
                 allow_ungrounded: bool = False,
                 allow_nonfree: bool = False,
                 batch_mode: str | None = None,
                 resume: bool = False) -> CreateResult:
    """``spools create``: the whole build behind one consent + one cost gate.

    Resolves any missing shas from the declared tags, PINS them back into the
    spools.yaml (TOFU; the consent prompt shows the resolved shas), then on
    consent: acquire → source add → discover/index → onboard (its own cost
    prompt) → pack build. ``phases`` overrides the default implementations.
    """
    name, version, data = _load_spoolfile(spoolfile_path)

    # Grounding gate (§18.1): a Spool must be SCIP-indexable. If the recipe
    # DECLARES a language with no registered indexer, refuse here — before any
    # fetch or cost — instead of building from the low-confidence fallback.
    if not allow_ungrounded:
        bad = [str(lang) for lang in (data.get('languages') or [])
               if not is_scip_eligible(lang)]
        if bad:
            raise SpoolError(
                f"spool {name!r} declares language(s) {bad} that Ariadne "
                f'cannot SCIP-index (no indexer) — a Spool needs code-tier '
                f'grounding, so it cannot be built effectively. Add an '
                f'indexer, drop them from the corpus, or pass '
                f'--allow-ungrounded for a docs-only pack.',
            )

    # Resolve missing shas IN MEMORY so the consent prompt can show them; the
    # write-back happens ONLY after consent (declining leaves the file alone).
    resolved_any = False
    for entry_name, entry in (data.get('corpus') or {}).items():
        entry = entry or {}
        if not entry.get('sha'):
            if not entry.get('tag'):
                raise SpoolError(
                    f"corpus entry '{entry_name}' has neither sha nor tag "
                    f'— nothing to pin',
                )
            entry['sha'] = _resolve_sha(entry['url'], entry['tag'])
            data['corpus'][entry_name] = entry
            resolved_any = True

    packfile = Packfile(
        runtime=str(data['runtime']),
        corpus={
            n: CorpusEntry(url=str(e['url']), sha=str(e['sha']),
                           tag=e.get('tag'))
            for n, e in data['corpus'].items()
        },
        certify=tuple(data.get('certify') or ()),
    )
    # On resume the corpus is already fetched (acquire reuses clones at the
    # pinned sha), so skip the fetch-consent. onboard_approve stays tied to
    # `approve`, so `--resume` alone still shows the now-smaller cost preview
    # while `--resume --yes` is fully unattended.
    result = acquire(packfile, dest_dir=dest_dir,
                     approve=(approve or resume), confirm=confirm)
    if not result.accepted:
        return CreateResult(accepted=False)

    if resolved_any:
        # Consent accepted -> pin the resolved shas back (generated file).
        Path(spoolfile_path).write_text(
            yaml.safe_dump(data, sort_keys=False), encoding='utf-8',
        )

    # Post-clone backstop (§18.1): even when the recipe declared no languages,
    # a corpus with NO SCIP-indexable source is refused here — before the paid
    # onboard — so an undeclared unsupported project can't slip to a hollow pack.
    if not allow_ungrounded:
        ungrounded = unsupported_corpus_language(dest_dir)
        if ungrounded:
            raise SpoolError(
                f'spool {name!r}: the fetched corpus is {ungrounded}, which '
                f'Ariadne cannot SCIP-index — refusing to build an '
                f'ungrounded pack. Pass --allow-ungrounded to override.',
            )

    # License-admission gate (§18.1): a spool ships DERIVED docs + a SCIP index
    # from its corpus, so the corpus must be under a license that permits
    # redistributing derived work. Source-available / proprietary / unrecognized
    # corpora are refused (fail-closed) unless the builder opts into a
    # local-only pack with --allow-nonfree.
    if not allow_nonfree:
        nonfree = nonfree_corpora(dest_dir)
        if nonfree:
            listed = ', '.join(
                f"{repo} ({lic or 'no recognized open-source license'})"
                for repo, _cat, lic in nonfree)
            raise SpoolError(
                f'spool {name!r}: corpus {listed} is not redistribution-safe '
                f'under an open-source license — a spool pack is meant to be '
                f'shared. Use an open-source upstream (e.g. OpenTofu instead of '
                f'BUSL Terraform), or pass --allow-nonfree for a local-only '
                f'build.',
            )

    from config import get_config
    phases = phases or _default_phases(
        batch_mode, onboard_approve=approve,
        spools_model=get_config().spools_model)
    print('▸ registering source…')
    phases['source_add'](name, str(dest_dir))
    print('▸ indexing corpus with SCIP…')
    phases['index'](name)
    print('▸ generating docs (onboard — cost preview follows)…')
    phases['onboard'](name)
    print('▸ clustering spool themes (spool:<name> partition)…')
    phases.get('theme', lambda _n: None)(name)
    print('▸ building spool pack…')
    phases['build'](
        source=name, version=version, runtime=packfile.runtime,
        certify=packfile.certify, source_root=dest_dir, out_path=out_path,
        taxonomy=tuple(data.get('taxonomy') or ()),
        corpus_shas={n: e.sha for n, e in packfile.corpus.items()},
        runtime_components=dict(data.get('runtime_components') or {}),
        surfaces={
            str(k): [str(s) for s in v]
            for k, v in (data.get('surfaces') or {}).items()
            if isinstance(v, list)},
    )
    return CreateResult(accepted=True, pack_path=str(out_path))


def _run_cli(*argv) -> None:
    """Drive an ariadne CLI command in-process; loud on nonzero exit."""
    import sys

    from cli.main import main as cli_main
    saved = sys.argv
    sys.argv = ['ariadne', *argv]
    try:
        code = cli_main()
    finally:
        sys.argv = saved
    if code:
        raise SpoolError(f"`ariadne {' '.join(argv)}` failed (exit {code})")


def _default_phases(batch_mode=None, onboard_approve=False,
                    spools_model=None) -> dict:
    """The real pipeline (operational session): heavy phases keep their own
    interactive gates (onboard shows the cost preview and prompts).
    ``batch_mode`` ('batch' | 'live' | None) pre-selects onboard's embedding
    mode; None leaves onboard's own live-vs-batch prompt intact.
    ``onboard_approve`` skips onboard's cost prompt (the unattended path).
    ``spools_model``, when set, is passed as ``--model`` so the build uses it.
    """
    def source_add(name, path):
        from config import get_config
        # Register the corpus with test dirs + Scala/Java test globs excluded,
        # staleness-exempt (a spool corpus is pinned/immutable), and — crucially
        # — with dependency detection OFF: a spool is an ISOLATED environment
        # plugin, never tied to another source via a `depends_on` edge.
        if not get_config().set_source_config(
                name, path=path,
                exclude_dirs=list(_SPOOL_IGNORE_TEST_DIRS),
                exclude=list(_SPOOL_IGNORE_TEST_GLOBS),
                ignore_staleness=True,
                skip_dependency_detection=True):
            raise SpoolError(f'could not persist source {name!r} to ariadne.yaml')

    def index(name):
        # --quiet: scip/pyright warnings are noise in the Spool scope.
        # --best-effort: one language's indexer failing (e.g. spark's JVM build)
        # must not sink the whole spool — index what can be, fail only if none.
        _run_cli('discover', '--source', name, '--quiet')
        _run_cli('index', '--source', name, '--quiet', '--best-effort')

    def onboard(name):
        extra = [f'--{batch_mode}'] if batch_mode else []
        if spools_model:
            extra += ['--model', spools_model]
        extra += ['--doc-types-off', ','.join(_SPOOL_DOC_TYPES_OFF)]
        if onboard_approve:
            extra.append('--approve')
        _run_cli('onboard', '--source', name, *extra)

    def build(*, source, version, runtime, certify, source_root, out_path,
              taxonomy=(), runtime_components=None, surfaces=None,
              corpus_shas=None):
        from config import get_config
        from library import Library
        from spool_pack import build_pack
        with Library(get_config().db_path) as library:
            build_pack(
                library, environment=source, version=version,
                target_runtime=runtime, certified_docs=certify,
                source_root=source_root, out_path=out_path, taxonomy=taxonomy,
                runtime_components=runtime_components,
                surfaces=surfaces,
                corpus_shas=corpus_shas,
            )

    def theme(name):
        # The spool's OWN theme pass: cluster the corpus into
        # spool:<name>-associated themes (build_spool_internal_themes) and
        # summarize the dirty ones. Distinct from onboard's base '' pass above
        # — corpus themes are tagged to the spool so they never occupy (or
        # leak from) the base partition. Uses the generate_themes summarizer
        # directly, so the general refresh_themes / `themes build` flow is
        # untouched.
        import asyncio

        from config import get_config
        from docgen.themes import generate_themes
        from library import Library
        from spools import build_spool_internal_themes
        from writer import LibraryWriter

        model = spools_model or get_config().model
        with Library(get_config().db_path) as library:
            if not build_spool_internal_themes(library, name, {name}):
                return

            async def _summarize():
                async with LibraryWriter(library) as writer:
                    await generate_themes(library, writer, model=model)

            asyncio.run(_summarize())

    return {'source_add': source_add, 'index': index, 'onboard': onboard,
            'theme': theme, 'build': build}


def consent_text(packfile: Packfile) -> str:
    """Exactly what will be fetched, shown BEFORE any acquisition."""
    lines = [
        f'The spool build (runtime {packfile.runtime}) requires fetching:',
    ]
    for name, entry in sorted(packfile.corpus.items()):
        tag = f'@{entry.tag}' if entry.tag else ''
        lines.append(f'  {name}{tag}  (sha {entry.sha[:12]})  from {entry.url}')
    lines.append(
        'Cost preview for indexing/generation follows at onboard '
        '(`ariadne dry-run` / the onboard prompt).',
    )
    return '\n'.join(lines)


def _git(cwd, *args) -> str:
    result = subprocess.run(
        ['git', *args], cwd=cwd, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise SpoolError(
            f"git {' '.join(args)} failed: {result.stderr.strip()}",
        )
    return result.stdout.strip()


def _clone_shallow(dest_dir, name, url, ref, *, on_files=None, total=0) -> None:
    """Fetch ONLY the pinned revision (``--depth 1`` at the tag/ref), never the
    whole history (apache/spark's is gigabytes). Local PATH clones ignore
    --depth; a real remote (or file://) honours it. ``total``/``on_files`` feed
    an optional progress bar (the count is fetched before the clone)."""
    _git(dest_dir, 'clone', '--depth', '1', '--branch', str(ref), '-q',
         str(url), name)


def _fetch_one(dest_dir, name, entry: CorpusEntry, *, on_files=None) -> None:
    """Fresh shallow fetch pinned to the exact sha: count the remote's files
    (progress total, best-effort) BEFORE the clone, clone at the tag, ensure the
    pinned sha is present + checked out, then strip ``.git`` and write the pin
    marker. Fail-closed: any failure removes the half-made clone and is loud."""
    dest_dir = Path(dest_dir)
    clone = dest_dir / name
    shutil.rmtree(clone, ignore_errors=True)
    total = _remote_file_count(entry.url, entry.sha)
    try:
        _clone_shallow(dest_dir, name, entry.url, entry.tag or entry.sha,
                       on_files=on_files, total=total)
        if _git(clone, 'rev-parse', 'HEAD') != entry.sha:
            # The pin differs from the tag's commit — fetch the exact sha.
            _git(clone, 'fetch', '--depth', '1', 'origin', entry.sha)
            _git(clone, 'checkout', '-q', entry.sha)
        head = _git(clone, 'rev-parse', 'HEAD')
        if head != entry.sha:
            raise SpoolError(
                f'HEAD {head} does not match the declared sha {entry.sha}',
            )
    except SpoolError as exc:
        shutil.rmtree(clone, ignore_errors=True)
        raise SpoolError(
            f"acquiring corpus '{name}' at sha {entry.sha[:12]} failed: {exc}",
        ) from exc
    shutil.rmtree(clone / '.git', ignore_errors=True)
    (clone / _CORPUS_SHA_MARKER).write_text(entry.sha + '\n', encoding='utf-8')


def acquire(packfile: Packfile, *, dest_dir, approve: bool,
            confirm=input) -> AcquireResult:
    """Consent-gated clone-at-SHA for every corpus entry.

    ``dest_dir`` is ariadne's OWN build workspace, so a leftover there is ours
    to manage, never a hard-fail: an existing checkout is REUSED at the pinned
    sha (marker match), ADOPTED if a pre-marker checkout sits at the pin,
    REFETCHED if it's ours but stale, or REPLACED (loud) if unrecognised.
    """
    if not approve:
        answer = confirm(f'{consent_text(packfile)}\nProceed? [y/N] ')
        if str(answer).strip().lower() not in ('y', 'yes'):
            return AcquireResult(accepted=False)

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    cloned = []
    for name, entry in sorted(packfile.corpus.items()):
        cloned.append(name)                             # every branch below keeps it
        clone = dest_dir / name
        marker = clone / _CORPUS_SHA_MARKER
        if clone.exists():
            if marker.exists():
                if marker.read_text().strip() == entry.sha:
                    continue                            # reuse (idempotent)
                _fetch_one(dest_dir, name, entry)       # ours, pin changed
                continue
            if (clone / '.git').exists():
                try:
                    head = _git(clone, 'rev-parse', 'HEAD')
                except SpoolError:
                    head = None
                if head == entry.sha:
                    # legacy pre-marker checkout at the pin → adopt in place
                    shutil.rmtree(clone / '.git', ignore_errors=True)
                    (clone / _CORPUS_SHA_MARKER).write_text(
                        entry.sha + '\n', encoding='utf-8')
                    continue
                _fetch_one(dest_dir, name, entry)       # ours, stale
                continue
            print(
                f'  {clone} is not a recognised ariadne corpus (no pin marker) '
                f'— replacing it (inside the {dest_dir} build workspace).')
            _fetch_one(dest_dir, name, entry)           # unrecognised → replace
            continue
        _fetch_one(dest_dir, name, entry)               # fresh
    return AcquireResult(accepted=True, cloned=tuple(cloned))
