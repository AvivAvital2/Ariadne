from __future__ import annotations

from slack_bridge.errors import to_user_message


def test_to_user_message_maps_errors_honestly():
    # Ambiguous/empty source: Ariadne's LookupError already names the configured
    # sources — surface them and ask which one.
    le = LookupError(
        "No source context — pass source= explicitly or run within a configured "
        "project tree. Currently configured sources: ['ariadne', 'projectb', 'projecta']."
    )
    msg = to_user_message(le)
    assert 'ariadne' in msg and 'projecta' in msg
    assert 'which' in msg.lower()

    # Timeout: honest, actionable — not a silent hang.
    tmsg = to_user_message(TimeoutError()).lower()
    assert 'too long' in tmsg or 'timed out' in tmsg or 'narrow' in tmsg

    # Generic tool error: quoted honestly, never masked.
    emsg = to_user_message(RuntimeError('skopeo not found'))
    assert 'skopeo not found' in emsg
    assert 'error' in emsg.lower()
