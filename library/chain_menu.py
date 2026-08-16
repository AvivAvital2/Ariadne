"""Offer the bundle before spending it: one line per thing the model could read.

Stage three used to hand synthesis the whole bundle. Measured at production width that is
2,645 hop lines and 883 descriptions — 240,945 tokens, $1.20 a question — of which 68% is
coordinates for hops the answer never mentions. A broad question doubles it.

So the chain is offered as a menu first. The model names what it wants, and only those
bodies are fetched: 973 definitions and ~1,060 sections costs about $0.13, and the second
call carries the handful that were chosen. The saving is not "titles are cheaper than
documents" — it is that a menu is **per symbol** while a chain is per occurrence, and a
``file:line`` is only needed for the hops an answer actually cites.

What this does *not* do is decide anything about the code. SCIP still decides what the
chain contains; the menu only lets the question influence which part of it is read, which
nothing in the walk can do — the walk has never seen the question. Selection is therefore
additive: a caller is free to send the structural spine as well, so a bad pick costs tokens
rather than evidence.

Two halves, labelled, because they are not equally reliable:

* **definitions** — the ``catalog`` entry for a symbol the walk reached, anchored to an
  exact ``file:line`` from the index. This is what an answer cites.
* **sections** — headings of the ``explanation`` documents covering the same files.
  Generated prose about a module, with no line-level anchor. Background, not citation.

Selection is by number. A number either labels something or it does not, so a model cannot
conjure a symbol by misspelling one and nothing here needs fuzzy matching; unknown numbers
are reported for the caller to see rather than interpreted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from library import Library
    from library.chain_bundle import BundleHop

#: How much of a description a menu line shows. A display preference, not a derivation:
#: enough to recognise what a definition is for, and no more, because the body is what the
#: second call fetches. Measured, the first line of a catalog description averages ~360
#: characters and the whole menu is 1,920 lines.
SUMMARY_CHARS = 90

#: ``1``/``12`` for a definition, ``S1``/``S12`` for a section, however the model writes it.
_CHOICE = re.compile(r'\b(S?)(\d{1,4})\b', re.IGNORECASE)


@dataclass(frozen=True)
class ChainMenu:
    """What the chain offers, and how to read a reply about it."""

    text: str = ''
    #: menu number -> qualified name
    symbols: dict = field(default_factory=dict)
    #: menu label (``S3``) -> ``(document_id, section idx)``
    sections: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Selection:
    """What the model asked for, resolved against the menu it was given."""

    symbols: list = field(default_factory=list)
    sections: list = field(default_factory=list)
    #: Labels that matched nothing. Reported, never guessed at.
    unknown: tuple = ()


def _owner_qualified(qualified_name: str) -> str:
    """``Classic.writeAllChanges`` — enough to choose between, without the full package.

    A bare last segment cannot be chosen between: a corpus has many ``apply`` and many
    ``write``. The owner is what disambiguates, and the full package name is what the
    second call fetches anyway.
    """
    parts = qualified_name.split('.')
    return '.'.join(parts[-2:]) if len(parts) > 1 else qualified_name


def _summary(content: str) -> str:
    """The sentence a choice is made on, out of a catalog body.

    A catalog document opens with kind, qualified name, package, path and signature, then
    carries ``Description: ...``. The menu line already prints the name, so the header adds
    only length: measured on the live store, taking the first line produced entries like
    ``catalog.CatalogManager — scala_object org.apache.spark.sql.connector.catalog…``, which
    says nothing the name had not. The description is the part that distinguishes.
    """
    lines = [line.strip() for line in (content or '').splitlines() if line.strip()]
    for line in lines:
        if line.lower().startswith('description:'):
            return line.split(':', 1)[1].strip()
    # No description: take the signature the header ends with (``… :: trait Foo``) rather
    # than the header itself, which carries the file path — the one thing a menu must not
    # spend, and what the end-to-end harness caught on its first run.
    for line in lines:
        if ' :: ' in line:
            return line.split(' :: ', 1)[1].strip()
    return lines[0] if lines else ''


def menu_for(library: 'Library', hops: list['BundleHop'], *, source: str) -> ChainMenu:
    """One line per definition the chain reached, then one per section covering its files."""
    definitions: dict[str, 'BundleHop'] = {}
    for hop in hops:
        definitions.setdefault(hop.citation.qualified_name, hop)
    if not definitions:
        return ChainMenu()

    files = sorted({hop.citation.file for hop in definitions.values()})
    sections: dict[str, tuple[str, int]] = {}
    section_lines: list[str] = []
    # A hop's ``evidence`` is what rationing allowed to travel as proof; ``plumbing`` and
    # ``revisit`` carry none by design. Choosing is a different job from evidencing, so the
    # menu reads the document itself — otherwise a quarter of the lines are bare names and
    # the model picks blind. The documents are already resolved; this only reads them.
    described: dict[str, str] = {}
    wanted_docs = sorted({hop.document_id for hop in definitions.values()
                          if hop.document_id})
    with library._conn_provider.acquire() as conn:
        for start in range(0, len(wanted_docs), 400):
            chunk = wanted_docs[start:start + 400]
            placeholders = ','.join('?' * len(chunk))
            for doc_id, content in conn.execute(
                    f'SELECT id, content FROM documents WHERE id IN ({placeholders})',
                    chunk):
                described[doc_id] = content or ''
        rows: list[tuple[str, str, int, str]] = []
        for start in range(0, len(files), 400):
            chunk = files[start:start + 400]
            like = ' OR '.join(['d.source_files LIKE ?'] * len(chunk))
            rows += conn.execute(
                f'SELECT d.id, d.title, s.idx, s.heading FROM sections s '
                f'JOIN documents d ON d.id = s.document_id '
                f"WHERE d.content_type = 'explanation' AND ({like}) "
                f'ORDER BY d.title, s.idx',
                [f'%{path}%' for path in chunk]).fetchall()
    for doc_id, title, idx, heading in rows:
        label = f'S{len(sections) + 1}'
        sections[label] = (doc_id, idx)
        section_lines.append(f'  {label}. {title} -> {heading}')

    symbols: dict[str, str] = {}
    definition_lines: list[str] = []
    for number, (qualified_name, hop) in enumerate(definitions.items(), start=1):
        symbols[str(number)] = qualified_name
        # No ``file:line`` here. A coordinate is what an answer *cites*, and the second call
        # carries it for the handful chosen; in the menu it is 87,000 characters of JVM path
        # across 973 lines, spent on a choice that is made by name and purpose.
        summary = _summary(described.get(hop.document_id or '', '') or hop.evidence or '')
        shown = summary[:SUMMARY_CHARS].rstrip()
        definition_lines.append(
            f'  {number}. {_owner_qualified(qualified_name)}'
            + (f' — {shown}{"…" if len(summary) > SUMMARY_CHARS else ""}' if shown else ''))

    text = '\n'.join([
        'DEFINITIONS the chain reached. Each has an exact file:line in the index, supplied '
        'with the body when you ask for it — these are what an answer cites.',
        *definition_lines,
    ] + ([
        '',
        'SECTIONS of the documents covering the same files. Generated prose about a module, '
        'with no line-level anchor: background only, never cited.',
        *section_lines,
    ] if section_lines else []))
    return ChainMenu(text=text, symbols=symbols, sections=sections)


def resolve_selection(menu: ChainMenu, reply: str) -> Selection:
    """The numbers in ``reply``, resolved against ``menu``. Order is the reply's order."""
    chosen_symbols: list[str] = []
    chosen_sections: list[tuple[str, int]] = []
    unknown: list[str] = []
    for prefix, digits in _CHOICE.findall(reply or ''):
        if prefix:
            label = f'S{digits}'
            target = menu.sections.get(label)
            if target is None:
                unknown.append(label)
            elif target not in chosen_sections:
                chosen_sections.append(target)
            continue
        qualified_name = menu.symbols.get(digits)
        if qualified_name is None:
            unknown.append(digits)
        elif qualified_name not in chosen_symbols:
            chosen_symbols.append(qualified_name)
    return Selection(symbols=chosen_symbols, sections=chosen_sections,
                     unknown=tuple(unknown))


@dataclass(frozen=True)
class Fetched:
    """The bodies the model asked for, and nothing else."""

    #: qualified name -> the ``catalog`` description
    definitions: dict = field(default_factory=dict)
    #: ``(document title, heading, content)`` for each chosen section
    sections: list = field(default_factory=list)


def fetch_selected(library: 'Library', selection: Selection,
                   hops: list['BundleHop']) -> Fetched:
    """Read only what was chosen.

    Document ids come from the hops rather than being recomputed, so this cannot disagree
    with what :func:`library.chain_bundle.curate_bundle` already resolved.
    """
    ids = {hop.citation.qualified_name: hop.document_id for hop in hops
           if hop.document_id}
    wanted = {name: ids[name] for name in selection.symbols if name in ids}
    definitions: dict[str, str] = {}
    sections: list[tuple[str, str, str]] = []
    if not wanted and not selection.sections:
        return Fetched()
    with library._conn_provider.acquire() as conn:
        if wanted:
            by_id = {doc_id: name for name, doc_id in wanted.items()}
            order = list(wanted)
            found: dict[str, str] = {}
            for start in range(0, len(by_id), 400):
                chunk = list(by_id)[start:start + 400]
                placeholders = ','.join('?' * len(chunk))
                for doc_id, content in conn.execute(
                        f'SELECT id, content FROM documents '
                        f'WHERE id IN ({placeholders})', chunk):
                    found[by_id[doc_id]] = content or ''
            definitions = {name: found[name] for name in order if name in found}
        for document_id, idx in selection.sections:
            row = conn.execute(
                'SELECT d.title, s.heading, s.content FROM sections s '
                'JOIN documents d ON d.id = s.document_id '
                'WHERE s.document_id = ? AND s.idx = ?', (document_id, idx)).fetchone()
            if row is not None:
                sections.append((row[0], row[1], row[2] or ''))
    return Fetched(definitions=definitions, sections=sections)


def render_selected(hops: list['BundleHop'], selection: Selection,
                    fetched: Fetched) -> str:
    """The chain restricted to what was chosen: execution order, coordinates, bodies.

    Execution order survives the restriction — the hops keep the order the walk produced,
    which is what makes a chain explicable. What was *not* chosen is counted rather than
    dropped in silence, so the model can see the chain was larger than what it asked for and
    say so if the answer needs more.
    """
    chosen = set(selection.symbols)
    lines: list[str] = []
    shown: set[str] = set()
    for hop in hops:
        name = hop.citation.qualified_name
        if name not in chosen:
            continue
        indent = '  ' * (hop.citation.hop - 1)
        site = ('referenced at' if hop.citation.relation == 'references'
                else 'called at')
        lines.append(
            f'{indent}{name}  [{hop.citation.file}:{hop.citation.line_start}]'
            f'  {site} {hop.citation.call_site_file}:{hop.citation.call_site_line}')
        body = fetched.definitions.get(name)
        if body and name not in shown:
            shown.add(name)
            lines.append(f'{indent}    {_summary(body)}')
    remaining = len({hop.citation.qualified_name for hop in hops}) - len(chosen)
    if remaining > 0:
        lines.append(f'... {remaining} further definition(s) in the chain were not '
                     f'requested; say so if the answer needs them.')
    if fetched.sections:
        lines.append('')
        lines.append('Background (module prose, no line-level anchor — do not cite):')
        for title, heading, content in fetched.sections:
            lines.append(f'  {title} -> {heading}')
            lines.append(f'    {content.strip()}')
    return '\n'.join(lines)
