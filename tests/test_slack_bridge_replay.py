from __future__ import annotations

from slack_bridge.replay import _to_transcript, load_thread


def test_to_transcript_maps_roles_drops_placeholders_keeps_order():
    bot = 'UBOT'
    messages = [
        {'user': 'UALICE', 'text': 'how does projecta export images?'},
        {'user': bot, 'text': '🔎 Searching the projecta docs…'},                 # placeholder → drop
        {'user': bot, 'text': 'In file-export mode it uses skopeo + pigz.'},  # answer
        {'user': 'UBOB', 'text': 'and the registry path?'},                   # 2nd human
        {'user': bot, 'text': '   '},                                         # empty → drop
        {'user': bot, 'text': 'It uses skopeo copy to docker://.'},
    ]

    turns = _to_transcript(messages, bot_user_id=bot)

    # Order preserved; placeholder + empty dropped.
    assert [t.role for t in turns] == ['user', 'assistant', 'user', 'assistant']
    # Human turns carry a speaker prefix (multi-human attribution).
    assert 'UALICE' in turns[0].text and 'export images' in turns[0].text
    assert 'UBOB' in turns[2].text
    # Bot turns map to assistant, verbatim.
    assert turns[1].text == 'In file-export mode it uses skopeo + pigz.'
    assert turns[3].text == 'It uses skopeo copy to docker://.'
    # No placeholder text leaks into the transcript.
    assert all('🔎' not in t.text for t in turns)


async def test_load_thread_surfaces_supported_thread_images():
    # The thread is the "correspondence": an image attached to an earlier
    # message must be discoverable on the cold path, junk filtered out.
    bot = 'UBOT'
    messages = [
        {'user': 'UALICE', 'text': 'look at this', 'files': [
            {'id': 'F1', 'mimetype': 'image/png', 'url_private': 'u1'},
            {'id': 'SKIP', 'mimetype': 'application/pdf', 'url_private': 'p'},
        ]},
        {'user': bot, 'text': 'on it'},
        {'user': 'UALICE', 'text': 'what is wrong?'},
    ]

    class _Slack:
        async def conversations_replies(self, channel, ts):  # noqa: ARG002
            return {'messages': messages}

    ctx = await load_thread(_Slack(), 'C', 'T', bot)

    assert [(r.id, r.media_type) for r in ctx.images] == [('F1', 'image/png')]
