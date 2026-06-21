"""Per-doc-type staleness — Tier 1 (detection API).

``stale_doc_types`` reports, per file, exactly the requested doc types that need
generating (the missing ones, or all of them when the source changed) — so
generation and pricing can do only those instead of regenerating present, valid
docs. See ``designs/per-doctype-staleness.md``. Evolving test: one growing
scenario refined slice by slice.
"""
from __future__ import annotations

import numpy as np

from docgen.staleness import StalenessTracker
from library import Library

_ALL = ('explanation', 'architecture', 'qa', 'gotcha', 'diagram')


def _seed_docs(lib, file_path, content_types, *, source_name='test'):
    """Register doc rows in the library for ``file_path``; return their IDs."""
    doc_ids = []
    for ct in content_types:
        doc = lib.add_document(
            content_type=ct,
            title=f'doc-{ct}',
            content='body',
            source_files=[str(file_path)],
            embedding=np.zeros(3072, dtype=np.float32),
            metadata={},
            source_name=source_name,
        )
        doc_ids.append(doc.id)
    return doc_ids


def test_stale_doc_types_returns_only_missing_types(tmp_path):
    """src_a has explanation+architecture; requesting all five reports ONLY the
    three missing types. A file already covering every requested type is
    omitted entirely (no work to do)."""
    src_a = tmp_path / 'src_a.py'
    src_a.write_text('a = 1\n', encoding='utf-8')
    src_b = tmp_path / 'src_b.py'
    src_b.write_text('b = 2\n', encoding='utf-8')

    lib = Library(tmp_path / 'lib.db')
    try:
        ids_a = _seed_docs(lib, src_a, ['explanation', 'architecture'])
        ids_b = _seed_docs(lib, src_b, ['explanation', 'architecture'])
        tracker = StalenessTracker(tmp_path / 'stale.db')
        try:
            tracker.record_documentation(src_a, ids_a, base_path=tmp_path)
            tracker.record_documentation(src_b, ids_b, base_path=tmp_path)

            # src_a: request all five → only the three missing come back.
            result = tracker.stale_doc_types(
                [src_a, src_b], base_path=tmp_path,
                requested_types=_ALL, library=lib,
            )
            assert set(result.get(src_a, ())) == {'qa', 'gotcha', 'diagram'}, (
                f'expected only the missing types; got {result.get(src_a)}'
            )

            # src_b: request only what it already has → omitted (no work).
            covered = tracker.stale_doc_types(
                [src_b], base_path=tmp_path,
                requested_types=('explanation', 'architecture'), library=lib,
            )
            assert src_b not in covered, (
                f'fully-covered file must be omitted; got {covered}'
            )
        finally:
            tracker.close()
    finally:
        lib.close()


def test_changed_source_marks_all_effective_types(tmp_path):
    """When the source file changed, *every* requested type is stale — not just
    the absent ones — because a content change invalidates the docs that exist."""
    src = tmp_path / 'src_c.py'
    src.write_text('c = 1\n', encoding='utf-8')

    lib = Library(tmp_path / 'lib.db')
    try:
        ids = _seed_docs(lib, src, ['explanation', 'architecture'])
        tracker = StalenessTracker(tmp_path / 'stale.db')
        try:
            tracker.record_documentation(src, ids, base_path=tmp_path)
            src.write_text('c = 999\n', encoding='utf-8')  # source changes → hash differs
            result = tracker.stale_doc_types(
                [src], base_path=tmp_path,
                requested_types=('explanation', 'architecture'), library=lib,
            )
            assert set(result.get(src, ())) == {'explanation', 'architecture'}, (
                f'changed source must mark all requested types stale; '
                f'got {result.get(src)}'
            )
        finally:
            tracker.close()
    finally:
        lib.close()


def test_missing_types_are_filtered_to_what_the_language_supports(tmp_path):
    """The reported types are intersected with LANGUAGE_DOC_TYPES — a JSON file,
    never documented, requesting all five yields only ``explanation`` (the sole
    type JSON supports), not five eternally-unsatisfiable types."""
    src = tmp_path / 'data.json'
    src.write_text('{}\n', encoding='utf-8')

    lib = Library(tmp_path / 'lib.db')
    try:
        tracker = StalenessTracker(tmp_path / 'stale.db')
        try:
            result = tracker.stale_doc_types(
                [src], base_path=tmp_path, requested_types=_ALL, library=lib,
            )
            assert set(result.get(src, ())) == {'explanation'}, (
                f'json supports only explanation; got {result.get(src)}'
            )
        finally:
            tracker.close()
    finally:
        lib.close()


def test_partial_recording_merges_doc_ids_for_idempotent_rerun(tmp_path):
    """Recording a partial generation MERGES the new doc_ids into the existing
    record (deterministic ids dedupe), instead of replacing — so the reused
    types aren't dropped and a re-run sees the file complete and skips it."""
    src = tmp_path / 'src_d.py'
    src.write_text('d = 1\n', encoding='utf-8')

    lib = Library(tmp_path / 'lib.db')
    try:
        ids = {ct: _seed_docs(lib, src, [ct])[0] for ct in _ALL}
        tracker = StalenessTracker(tmp_path / 'stale.db')
        try:
            # First pass documented only explanation + architecture.
            tracker.record_documentation(
                src, [ids['explanation'], ids['architecture']], base_path=tmp_path,
            )
            # A partial generation then records ONLY the three that were missing.
            tracker.record_documentation(
                src, [ids['qa'], ids['gotcha'], ids['diagram']], base_path=tmp_path,
            )
            rec = tracker.get_record(str(src.relative_to(tmp_path)))
            assert set(rec.doc_ids) == set(ids.values()), (
                f'partial recording must merge with the existing doc_ids '
                f'(not replace); got {rec.doc_ids}'
            )
            # Re-run requesting all five → nothing to do (idempotent).
            result = tracker.stale_doc_types(
                [src], base_path=tmp_path, requested_types=_ALL, library=lib,
            )
            assert src not in result, (
                f're-run after the fill must skip the now-complete file; got {result}'
            )
        finally:
            tracker.close()
    finally:
        lib.close()


def test_library_none_falls_back_to_hash_only(tmp_path):
    """Without a ``library`` to resolve existing types, a documented file whose
    source is unchanged is NOT stale (legacy hash-only, mirroring
    ``get_stale_files``) — it must not be reported as needing every type."""
    src = tmp_path / 'src_e.py'
    src.write_text('e = 1\n', encoding='utf-8')

    tracker = StalenessTracker(tmp_path / 'stale.db')
    try:
        tracker.record_documentation(src, ['some-doc-id'], base_path=tmp_path)
        result = tracker.stale_doc_types(
            [src], base_path=tmp_path, requested_types=_ALL, library=None,
        )
        assert src not in result, (
            f'a recorded, unchanged file must not be stale without a library; '
            f'got {result}'
        )
    finally:
        tracker.close()
def test_get_stale_files_legacy_returns_only_hash_stale(tmp_path):
    """With no ``requested_types`` (legacy mode), ``get_stale_files`` returns
    exactly the hash-stale files — never-documented or content-changed — and
    omits a documented, unchanged file. This is the contract ``stale_doc_types``
    cannot reproduce (it returns ``{}`` when no types are requested), so the
    legacy path must stay distinct from any type-aware delegation."""
    documented = tmp_path / 'src_documented.py'
    documented.write_text('keep = 1\n', encoding='utf-8')
    changed = tmp_path / 'src_changed.py'
    changed.write_text('was = 1\n', encoding='utf-8')
    never = tmp_path / 'src_never.py'
    never.write_text('fresh = 1\n', encoding='utf-8')

    lib = Library(tmp_path / 'lib.db')
    try:
        ids_doc = _seed_docs(lib, documented, ['explanation'])
        ids_chg = _seed_docs(lib, changed, ['explanation'])
        tracker = StalenessTracker(tmp_path / 'stale.db')
        try:
            tracker.record_documentation(documented, ids_doc, base_path=tmp_path)
            tracker.record_documentation(changed, ids_chg, base_path=tmp_path)
            changed.write_text('was = 999\n', encoding='utf-8')  # hash now differs

            stale = tracker.get_stale_files([documented, changed, never], base_path=tmp_path)
            assert set(stale) == {changed, never}, (
                f'legacy mode returns only hash-stale files; got {stale}'
            )
        finally:
            tracker.close()
    finally:
        lib.close()


def test_get_stale_files_type_aware_agrees_with_stale_doc_types_keys(tmp_path):
    """In type-aware mode (requested_types + library), absent exempt/empty edge
    cases, the files ``get_stale_files`` flags are exactly the keys of
    ``stale_doc_types`` — the relationship that lets them share one
    existing-types primitive. Covers fully-covered (skipped), partially-covered
    (flagged), and never-documented (flagged)."""
    covered = tmp_path / 'src_covered.py'
    covered.write_text('c = 1\n', encoding='utf-8')
    partial = tmp_path / 'src_partial.py'
    partial.write_text('p = 1\n', encoding='utf-8')
    never = tmp_path / 'src_new.py'
    never.write_text('n = 1\n', encoding='utf-8')

    lib = Library(tmp_path / 'lib.db')
    try:
        ids_cov = _seed_docs(lib, covered, _ALL)
        ids_par = _seed_docs(lib, partial, ['explanation', 'architecture'])
        tracker = StalenessTracker(tmp_path / 'stale.db')
        try:
            tracker.record_documentation(covered, ids_cov, base_path=tmp_path)
            tracker.record_documentation(partial, ids_par, base_path=tmp_path)

            paths = [covered, partial, never]
            files = tracker.get_stale_files(
                paths, base_path=tmp_path, requested_types=_ALL, library=lib,
            )
            types = tracker.stale_doc_types(
                paths, base_path=tmp_path, requested_types=_ALL, library=lib,
            )
            assert set(files) == set(types.keys()) == {partial, never}, (
                f'type-aware get_stale_files keys must match stale_doc_types; '
                f'files={files} types={list(types)}'
            )
        finally:
            tracker.close()
    finally:
        lib.close()


def test_get_stale_files_flags_never_documented_exempt_as_coverage_gap(tmp_path):
    """A never-documented file that is staleness-EXEMPT is still returned by
    ``get_stale_files``: exemption suppresses only the content-changed nag, but
    a never-documented file is a coverage gap that must surface (the
    ``ignore_staleness`` contract). This is the behavior a naive
    ``list(stale_doc_types(...).keys())`` wrapper would silently drop, since
    ``stale_doc_types`` skips exempt paths before the never-documented check."""
    never = tmp_path / 'src_vendored.py'
    never.write_text('v = 1\n', encoding='utf-8')

    def is_exempt(rel_path):
        return rel_path == 'src_vendored.py'

    tracker = StalenessTracker(tmp_path / 'stale.db')
    try:
        stale = tracker.get_stale_files([never], base_path=tmp_path, is_exempt=is_exempt)
        assert never in stale, (
            f'never-documented exempt file is a coverage gap and must surface; got {stale}'
        )
    finally:
        tracker.close()
def test_coverage_gaps_reports_missing_types_without_hashing(tmp_path):
    """``coverage_gaps`` reports files MISSING a requested doc type — never
    documented (all effective types) or recorded-but-incomplete (the absent
    ones) — and IGNORES content staleness (it does not hash): a documented file
    whose source changed but still has every requested type is NOT a coverage
    gap (the commit-diff gate owns content changes). Fully-covered files are
    omitted."""
    covered = tmp_path / 'src_covered.py'
    covered.write_text('c = 1\n', encoding='utf-8')
    partial = tmp_path / 'src_partial.py'
    partial.write_text('p = 1\n', encoding='utf-8')
    changed = tmp_path / 'src_changed.py'
    changed.write_text('h = 1\n', encoding='utf-8')
    never = tmp_path / 'src_never.py'
    never.write_text('n = 1\n', encoding='utf-8')

    lib = Library(tmp_path / 'lib.db')
    try:
        ids_cov = _seed_docs(lib, covered, ['explanation', 'architecture'])
        ids_par = _seed_docs(lib, partial, ['explanation'])
        ids_chg = _seed_docs(lib, changed, ['explanation', 'architecture'])
        tracker = StalenessTracker(tmp_path / 'stale.db')
        try:
            tracker.record_documentation(covered, ids_cov, base_path=tmp_path)
            tracker.record_documentation(partial, ids_par, base_path=tmp_path)
            tracker.record_documentation(changed, ids_chg, base_path=tmp_path)
            changed.write_text('h = 999\n', encoding='utf-8')  # content changes (hash differs)

            gaps = tracker.coverage_gaps(
                [covered, partial, changed, never], base_path=tmp_path,
                requested_types=('explanation', 'architecture'), library=lib,
            )
            assert set(gaps.get(partial, ())) == {'architecture'}, gaps
            assert set(gaps.get(never, ())) == {'explanation', 'architecture'}, gaps
            assert covered not in gaps, 'fully-covered file must be omitted'
            assert changed not in gaps, (
                'coverage_gaps must ignore content staleness (no hashing); the '
                'commit-diff gate handles changed sources'
            )
        finally:
            tracker.close()
    finally:
        lib.close()


def test_files_for_generation_commit_gate_unions_changed_and_coverage_gaps(tmp_path):
    """The commit-diff gate now UNIONS the changed files with files missing a
    requested doc type. A synced source picks up newly-requested types without
    --force: changed files regenerate the FULL requested set (omitted from the
    narrowing map), coverage-gap files only their MISSING types. Evolves to the
    exact sampleproj case — an EMPTY restrict (nothing changed) still fills gaps."""
    changed = tmp_path / 'src_changed.py'
    changed.write_text('c = 1\n', encoding='utf-8')
    gap = tmp_path / 'src_gap.py'
    gap.write_text('g = 1\n', encoding='utf-8')
    covered = tmp_path / 'src_covered.py'
    covered.write_text('k = 1\n', encoding='utf-8')

    lib = Library(tmp_path / 'lib.db')
    try:
        ids_gap = _seed_docs(lib, gap, ['explanation'])
        ids_cov = _seed_docs(lib, covered, ['explanation', 'architecture'])
        tracker = StalenessTracker(tmp_path / 'stale.db')
        try:
            tracker.record_documentation(gap, ids_gap, base_path=tmp_path)
            tracker.record_documentation(covered, ids_cov, base_path=tmp_path)

            files, doc_types = tracker.files_for_generation(
                [changed, gap, covered], base_path=tmp_path,
                requested_types=('explanation', 'architecture'), library=lib,
                is_exempt=None, restrict_to_files=frozenset({'src_changed.py'}),
                force=False,
            )
            assert set(files) == {changed, gap}, (
                f'changed file + coverage gap selected, covered file skipped; '
                f'got {sorted(p.name for p in files)}'
            )
            assert set(doc_types.get(gap, ())) == {'architecture'}, (
                f'coverage-gap file regenerates only its missing type; got {doc_types.get(gap)}'
            )
            assert changed not in doc_types, (
                'a source-changed file regenerates the full requested set'
            )

            # Exact sampleproj case: nothing changed (empty restrict) still fills gaps.
            files2, doc_types2 = tracker.files_for_generation(
                [gap, covered], base_path=tmp_path,
                requested_types=('explanation', 'architecture'), library=lib,
                is_exempt=None, restrict_to_files=frozenset(), force=False,
            )
            assert files2 == [gap], (
                f'empty restrict must still select coverage gaps, not skip all; got {files2}'
            )
            assert set(doc_types2.get(gap, ())) == {'architecture'}
        finally:
            tracker.close()
    finally:
        lib.close()


def test_files_for_generation_force_and_no_restrict_branches(tmp_path):
    """force=True → every file, full set ({} narrowing). restrict_to_files=None
    → the full type-aware staleness pass."""
    gap = tmp_path / 'src_gap.py'
    gap.write_text('g = 1\n', encoding='utf-8')
    other = tmp_path / 'src_other.py'
    other.write_text('o = 1\n', encoding='utf-8')

    lib = Library(tmp_path / 'lib.db')
    try:
        ids_gap = _seed_docs(lib, gap, ['explanation'])
        tracker = StalenessTracker(tmp_path / 'stale.db')
        try:
            tracker.record_documentation(gap, ids_gap, base_path=tmp_path)

            forced, forced_types = tracker.files_for_generation(
                [gap, other], base_path=tmp_path,
                requested_types=('explanation', 'architecture'), library=lib,
                is_exempt=None, restrict_to_files=None, force=True,
            )
            assert set(forced) == {gap, other} and forced_types == {}, (
                'force regenerates every file at the full requested set'
            )

            stale, stale_types = tracker.files_for_generation(
                [gap, other], base_path=tmp_path,
                requested_types=('explanation', 'architecture'), library=lib,
                is_exempt=None, restrict_to_files=None, force=False,
            )
            # gap: missing architecture; other: never documented -> both stale.
            assert set(stale) == {gap, other}
            assert set(stale_types.get(gap, ())) == {'architecture'}
        finally:
            tracker.close()
    finally:
        lib.close()
def test_exempt_suppresses_content_change_but_surfaces_coverage_gaps(tmp_path):
    """Corrected ignore_staleness semantics: exemption suppresses only the
    content-CHANGED signal. A file MISSING a requested doc type is a coverage gap
    that still surfaces — in stale_doc_types, get_stale_files, AND under the
    commit-diff gate (files_for_generation). A complete-but-content-changed exempt
    file is suppressed; a partially-documented exempt file surfaces its missing
    types. A missing doc is a missing doc."""
    complete = tmp_path / 'src_complete.py'   # all requested types; source then changes
    complete.write_text('c = 1\n', encoding='utf-8')
    partial = tmp_path / 'src_partial.py'     # missing architecture; unchanged
    partial.write_text('p = 1\n', encoding='utf-8')

    def exempt(rel):
        return True  # whole source staleness-exempt

    missing = {'architecture', 'qa', 'gotcha', 'diagram'}

    lib = Library(tmp_path / 'lib.db')
    try:
        ids_c = _seed_docs(lib, complete, _ALL)
        ids_p = _seed_docs(lib, partial, ['explanation'])
        tracker = StalenessTracker(tmp_path / 'stale.db')
        try:
            tracker.record_documentation(complete, ids_c, base_path=tmp_path)
            tracker.record_documentation(partial, ids_p, base_path=tmp_path)
            complete.write_text('c = 999\n', encoding='utf-8')  # content changed

            sdt = tracker.stale_doc_types(
                [complete, partial], base_path=tmp_path,
                requested_types=_ALL, library=lib, is_exempt=exempt,
            )
            assert complete not in sdt, 'content-change on a complete exempt file is suppressed'
            assert set(sdt.get(partial, ())) == missing, (
                f'coverage gap surfaces despite exemption; got {sdt.get(partial)}'
            )

            flagged = tracker.get_stale_files(
                [complete, partial], base_path=tmp_path,
                requested_types=_ALL, library=lib, is_exempt=exempt,
            )
            assert partial in flagged and complete not in flagged, (
                f'get_stale_files: exempt partial flagged, exempt complete-changed not; '
                f'got {[p.name for p in flagged]}'
            )

            files, doc_types = tracker.files_for_generation(
                [complete, partial], base_path=tmp_path,
                requested_types=_ALL, library=lib, is_exempt=exempt,
                restrict_to_files=frozenset(), force=False,
            )
            assert partial in files and complete not in files, (
                f'commit gate on an exempt source still fills coverage gaps; '
                f'got {[p.name for p in files]}'
            )
            assert set(doc_types.get(partial, ())) == missing
        finally:
            tracker.close()
    finally:
        lib.close()
def test_per_language_override_caps_effective_types(tmp_path):
    """A tracker built with a ``doc_types_by_language`` override caps the
    effective set per language. A never-documented python file requesting
    explanation+architecture, under an override of ('explanation',) for python,
    is stale for explanation ONLY — architecture is excluded — so the gate and
    estimate never flag (or price) a format the doc-type screen excluded.
    Without the override the same file is stale for both."""
    src = tmp_path / 'src_a.py'
    src.write_text('a = 1\n', encoding='utf-8')
    lib = Library(tmp_path / 'lib.db')
    try:
        requested = ('explanation', 'architecture')
        # Baseline: no override → a never-documented python file is stale for both.
        plain = StalenessTracker(tmp_path / 'plain.db')
        try:
            assert set(plain.stale_doc_types(
                [src], base_path=tmp_path, requested_types=requested, library=lib,
            ).get(src, ())) == {'explanation', 'architecture'}
        finally:
            plain.close()
        # Override caps python to explanation → architecture excluded.
        capped = StalenessTracker(
            tmp_path / 'capped.db',
            doc_types_by_language={'python': ('explanation',)},
        )
        try:
            assert set(capped.stale_doc_types(
                [src], base_path=tmp_path, requested_types=requested, library=lib,
            ).get(src, ())) == {'explanation'}, 'override must exclude architecture'
        finally:
            capped.close()
    finally:
        lib.close()
