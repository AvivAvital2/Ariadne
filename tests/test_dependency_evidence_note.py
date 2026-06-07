"""The dependency-detection approval panel must explain when its evidence
comes from a directory excluded from documentation (e.g. ``.venv/``).

Import analysis intentionally scans those dirs — locally, at no LLM cost — so
it can catch dependencies that surface only through an installed package. The
note therefore appears for excluded-dir evidence and is absent for first-party
imports.
"""
from cli.generate import _excluded_evidence_note
from docgen.dependency import DetectedDependency, ImportEvidence


def _dep(file_path: str) -> DetectedDependency:
    return DetectedDependency(
        source_name='auth-app',
        evidence=(
            ImportEvidence(
                file_path=file_path,
                line_number=26,
                import_statement='from auth_app.exceptions import X',
                source_name='auth-app',
            ),
        ),
    )


def test_excluded_evidence_note_only_for_excluded_dir_evidence():
    excluded = {'.venv', 'site-packages', '__pycache__'}

    # Evidence inside an excluded dir -> explanatory, cost-reassuring note.
    note = _excluded_evidence_note(
        _dep('be/.venv/lib/python3.9/site-packages/auth_app/uma_permissions.py'),
        excluded,
    )
    assert note, 'expected a note when evidence lives inside an excluded dir'
    assert '.venv' in note
    assert 'no' in note.lower() and 'cost' in note.lower(), 'must convey it is free'

    # First-party evidence -> no note at all.
    assert _excluded_evidence_note(_dep('be/app/services/auth.py'), excluded) == ''
