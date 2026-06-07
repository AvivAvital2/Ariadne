"""Dependency detection must let callers exclude sources from being targets.

Ariadne's own source is never a genuine dependency for a documented project —
and its name collides with common packages (e.g. the ``ariadne`` GraphQL
library) — so the caller passes it via ``ignore`` and the detector must drop it
while still reporting genuine dependencies.
"""
from docgen.dependency import detect_dependencies


def test_detect_dependencies_honors_ignore(tmp_path):
    proj = tmp_path / 'proj'
    proj.mkdir()
    (proj / 'app.py').write_text(
        'import ariadne\nfrom auth_app import AuthAdmin\n'
    )
    known = {'ariadne': tmp_path / 'ariadne', 'auth-app': tmp_path / 'auth_app'}
    for p in known.values():
        p.mkdir()

    # Baseline: both imports resolve to a known source.
    assert {d.source_name for d in detect_dependencies(proj, known)} == {
        'ariadne',
        'auth-app',
    }

    # Ignoring 'ariadne' (the tool's own source) drops it, keeps the rest.
    ignored = detect_dependencies(proj, known, ignore=frozenset({'ariadne'}))
    assert {d.source_name for d in ignored} == {'auth-app'}
