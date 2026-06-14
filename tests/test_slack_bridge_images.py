from __future__ import annotations

from slack_bridge.images import (
    MAX_IMAGE_BYTES,
    MAX_IMAGES,
    ImageRef,
    download_images,
    image_files_in,
)
from tests._slack_bridge_helpers import PLACEHOLDER


def _file(fid, mimetype='image/png', url='https://files.slack/x'):
    return {'id': fid, 'mimetype': mimetype, 'url_private': url}


def test_image_files_in_extracts_supported_filters_junk_and_dedupes():
    messages = [
        {'text': 'first', 'files': [
            _file('F1', 'image/png', 'u1'),
            _file('SKIP', 'application/pdf', 'p'),      # unsupported type
            {'id': 'NOURL', 'mimetype': 'image/png'},   # no url_private
        ]},
        {'text': 'a message with no files at all'},
        {'text': 'again', 'files': [
            _file('F1', 'image/png', 'u1-dup'),         # dup id -> first wins
            _file('F2', 'image/gif', 'u2'),
        ]},
    ]
    refs = image_files_in(messages)
    assert [(r.id, r.media_type, r.url) for r in refs] == [
        ('F1', 'image/png', 'u1'),
        ('F2', 'image/gif', 'u2'),
    ]
    assert all(isinstance(r, ImageRef) for r in refs)


def test_image_files_in_caps_at_max_images():
    files = [_file(f'F{i}') for i in range(MAX_IMAGES + 5)]
    refs = image_files_in([{'text': 'lots', 'files': files}])
    assert len(refs) == MAX_IMAGES
    assert [r.id for r in refs] == [f'F{i}' for i in range(MAX_IMAGES)]


async def test_download_images_downloads_skips_failures_and_oversize():
    refs = [
        ImageRef('OK', 'u-ok', 'image/png'),
        ImageRef('FAIL', 'u-fail', 'image/png'),     # fetch raises -> skipped
        ImageRef('BIG', 'u-big', 'image/jpeg'),       # oversize -> skipped
    ]

    async def fake_fetch(url, token):
        assert token == PLACEHOLDER                   # bot token is passed through
        if url == 'u-fail':
            raise RuntimeError('HTTP 404')
        if url == 'u-big':
            return b'x' * (MAX_IMAGE_BYTES + 1)
        return b'PNGBYTES'

    blobs = await download_images(refs, token=PLACEHOLDER, fetch=fake_fetch)

    assert [(b.media_type, b.data) for b in blobs] == [('image/png', b'PNGBYTES')]
