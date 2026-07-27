"""Builder acquisition with consent-before-fetch (slice b3).

A packfile declares the corpus a Spool build fetches: repo URL + tag +
**SHA** per entry (tags are mutable; the SHA is the pin). Before ANY
acquisition the consent prompt shows exactly what will be fetched —
the user requirement of 2026-07-08 — and ``--approve`` skips it for CI.
Clones are verified at the declared SHA; a mismatch fails loud and the
clone is removed. Runs on the BUILDER's machine only (§5: consumers
receive pure data). Design: IMPLEMENT.md (b3).
"""
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from spools import (SpoolError, is_scip_eligible, load_yaml_mapping, unsupported_corpus_language)


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


_RECIPES_DIR = Path(__file__).parent / 'spool_content' / 'recipes'


def _ask(prompt, label: str, default: str) -> str:
    """Prompt with a default shown in ``[...]``; an empty answer keeps it."""
    suffix = f' [{default}]' if default else ''
    resp = prompt(f'{label}{suffix}: ').strip()
    return resp or default


def setup_recipe(environment=None, *, out_path='spools.yaml', prompt=None,
                 available=None) -> Path:
    """Interactive recipe setup for the unified ``spools create``.

    The repo SET comes from the environment's built-in recipe (provided
    automatically); the user specifies each version. An existing
    ``out_path`` seeds the defaults, so re-running confirms/edits it
    rather than starting over. Returns the written path.
    """
    import yaml

    prompt = prompt or input
    out_path = Path(out_path)
    if available is None:
        available = sorted(p.stem for p in _RECIPES_DIR.glob('*.yaml'))
    if not available:
        raise SpoolError('no built-in spool recipes available')

    if environment is None:
        environment = _ask(prompt, f'Which spool? {available}', available[0])
    recipe_file = _RECIPES_DIR / f'{environment}.yaml'
    if not recipe_file.exists():
        raise SpoolError(
            f'no built-in recipe for {environment!r}; known environments: '
            f'{available}',
        )

    # An existing recipe seeds the defaults; otherwise the built-in template.
    base = yaml.safe_load(recipe_file.read_text(encoding='utf-8'))
    if out_path.exists():
        existing = yaml.safe_load(out_path.read_text(encoding='utf-8')) or {}
        if existing.get('corpus'):
            base = existing

    base['runtime'] = _ask(prompt, 'Runtime edition', base.get('runtime', ''))
    for repo, spec in base.get('corpus', {}).items():
        spec['tag'] = _ask(prompt, f'{repo} tag', spec.get('tag', ''))

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

    Annotated tags (every real release tag) list twice: the tag object at
    ``refs/tags/T`` and the peeled commit at ``refs/tags/T^{}`` — and
    ls-remote sorts the tag object first. Preference order here: peeled
    commit > lightweight tag > branch; never the tag object (checkout
    verification compares against HEAD, which is always a commit).
    """
    # peeled commit > lightweight tag > branch (never the tag object)
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


@dataclass(frozen=True)
class CreateResult:
    accepted: bool
    pack_path: str | None = None


def create_spool(spoolfile_path, *, dest_dir, out_path, approve: bool,
                 confirm=input, phases: dict | None = None,
                 allow_ungrounded: bool = False, batch_mode: str | None = None) -> CreateResult:
    """``spools create``: the whole build behind one consent + one cost gate.

    Resolves any missing shas from the declared tags, PINS them back into
    the spools.yaml (trust-on-first-use; the consent prompt shows the
    resolved shas), then on consent: acquire → source add → discover/index
    → onboard (its own cost prompt) → pack build. ``phases`` overrides the
    default implementations (tests inject fakes; the paid phases only run
    for real in the operational session).
    """
    name, version, data = _load_spoolfile(spoolfile_path)

    # Grounding gate (§18.1): a Spool must be SCIP-indexable. If the recipe
    # DECLARES a language with no registered indexer (e.g. Go), refuse here —
    # before any fetch or cost — instead of silently building from the
    # low-confidence raw-file/ast-grep fallback. --allow-ungrounded overrides.
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

    # Resolve any missing shas IN MEMORY so the consent prompt can show
    # them (HIGH-4). The write-back to disk happens ONLY after consent is
    # accepted — declining leaves the user's spools.yaml untouched.
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
    result = acquire(packfile, dest_dir=dest_dir, approve=approve,
                     confirm=confirm)
    if not result.accepted:
        return CreateResult(accepted=False)

    if resolved_any:
        # Consent accepted -> pin the resolved shas back (generated file;
        # PyYAML rewrite drops comments, same convention as ariadne.yaml).
        Path(spoolfile_path).write_text(
            yaml.safe_dump(data, sort_keys=False), encoding='utf-8',
        )

    # Post-clone backstop (§18.1): even when the recipe declared no languages,
    # a corpus that turns out to have NO SCIP-indexable source is refused here
    # — before the paid onboard — so an undeclared Go project can't slip
    # through to a hollow pack. --allow-ungrounded overrides.
    if not allow_ungrounded:
        ungrounded = unsupported_corpus_language(dest_dir)
        if ungrounded:
            raise SpoolError(
                f'spool {name!r}: the fetched corpus is {ungrounded}, which '
                f'Ariadne cannot SCIP-index — refusing to build an '
                f'ungrounded pack. Pass --allow-ungrounded to override.',
            )

    phases = phases or _default_phases(batch_mode, onboard_approve = approve)
    phases['source_add'](name, str(dest_dir))
    phases['index'](name)
    phases['onboard'](name)
    phases['build'](
        source=name, version=version, runtime=packfile.runtime,
        certify=packfile.certify, source_root=dest_dir, out_path=out_path,
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


def _default_phases(batch_mode=None, onboard_approve=False) -> dict:
    """The real pipeline (operational session): heavy phases keep their
    own interactive gates (onboard shows the cost preview and prompts).
    ``batch_mode`` ('batch' | 'live' | None) pre-selects onboard's
    embedding mode; None leaves onboard's own live-vs-batch prompt (the
    interactive toggle) intact. ``onboard_approve`` skips onboard's
    cost prompt (the unattended path).
    """
    def source_add(name, path):
        from config import get_config
        if not get_config().set_source_config(name, path=path):
            raise SpoolError(f'could not persist source {name!r} to ariadne.yaml')

    def index(name):
        _run_cli('discover', '--source', name)
        _run_cli('index', '--source', name)

    def onboard(name):
        extra = [f'--{batch_mode}'] if batch_mode else []
        if onboard_approve:
            extra.append('--approve')
        _run_cli('onboard', '--source', name, *extra)

    def build(*, source, version, runtime, certify, source_root, out_path):
        from config import get_config
        from library import Library
        from spool_pack import build_pack
        with Library(get_config().db_path) as library:
            build_pack(
                library, environment=source, version=version,
                target_runtime=runtime, certified_docs=certify,
                source_root=source_root, out_path=out_path,
            )

    return {'source_add': source_add, 'index': index,
            'onboard': onboard, 'build': build}


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


def acquire(packfile: Packfile, *, dest_dir, approve: bool,
            confirm=input) -> AcquireResult:
    """Consent-gated clone-at-SHA for every corpus entry.

    Declined consent acquires nothing. Each clone is checked out at the
    declared SHA and verified; on failure the clone is removed and the
    error is loud (fail-closed, no debris).
    """
    if not approve:
        answer = confirm(f'{consent_text(packfile)}\nProceed? [y/N] ')
        if str(answer).strip().lower() not in ('y', 'yes'):
            return AcquireResult(accepted=False)

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    cloned = []
    for name, entry in sorted(packfile.corpus.items()):
        clone = dest_dir / name
        if clone.exists():
            # CRIT-2: NEVER delete a pre-existing checkout. At the pinned
            # sha -> reuse (idempotent re-run/resume); anything else ->
            # refuse loudly and leave it exactly as found.
            try:
                head = _git(clone, 'rev-parse', 'HEAD')
            except SpoolError as exc:
                raise SpoolError(
                    f"corpus '{name}': {clone} exists but is not a git "
                    f'checkout ({exc}) — refusing to touch it; remove it '
                    f'or use a different --dest',
                ) from exc
            if head != entry.sha:
                raise SpoolError(
                    f"corpus '{name}': {clone} exists at {head[:12]}, "
                    f'pinned sha is {entry.sha[:12]} — refusing to touch '
                    f'it; remove it or use a different --dest',
                )
            cloned.append(name)
            continue
        try:
            _git(dest_dir, 'clone', '-q', entry.url, name)
            _git(clone, 'checkout', '-q', entry.sha)
            head = _git(clone, 'rev-parse', 'HEAD')
            if head != entry.sha:
                raise SpoolError(
                    f"corpus '{name}': HEAD {head} does not match the "
                    f'declared sha {entry.sha}',
                )
        except SpoolError as exc:
            # Cleanup is safe here: this clone was created by THIS run.
            shutil.rmtree(clone, ignore_errors=True)
            raise SpoolError(
                f"acquiring corpus '{name}' at sha {entry.sha[:12]} "
                f'failed: {exc}',
            ) from exc
        cloned.append(name)
    return AcquireResult(accepted=True, cloned=tuple(cloned))
