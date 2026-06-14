"""Incoming image support for the Slack bridge.

Slack delivers attached images as ``files[]`` entries on a message (with a
``url_private`` that needs the bot token to fetch). This module turns those
into :class:`ImageBlob` objects the agent runner hands to the model as vision
content blocks. Kept separate from transport/handler code so extraction,
download, and block-building are each unit-testable without Slack or the SDK.
"""
from __future__ import annotations

import base64
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from attrs import frozen

_logger = logging.getLogger(__name__)

# The image media types Claude vision accepts. Anything else (e.g. an svg or a
# pdf attached as a "file") is dropped before download — we never send the
# model bytes it can't decode.
SUPPORTED_MEDIA_TYPES: frozenset[str] = frozenset(
    {'image/jpeg', 'image/png', 'image/gif', 'image/webp'}
)

# Cap on images sent in a single turn. A thread can accumulate many screenshots;
# beyond this we keep the earliest (the ones most likely being referenced) and
# drop the rest rather than blow up the context / the upload.
MAX_IMAGES = 8

# Per-image byte ceiling (Anthropic rejects images above ~5 MB). Oversize
# attachments are skipped rather than sent and bounced.
MAX_IMAGE_BYTES = 5 * 1024 * 1024


@frozen
class ImageBlob:
    """A downloaded image, ready to send to the model as a content block."""

    media_type: str
    data: bytes

    def to_content_block(self) -> dict:
        """Render as an Anthropic ``image`` content block (base64 source).

        This is the format the Claude Code CLI passes through to the API; the
        SDK's stream-json input accepts it inside a user message's ``content``
        list (the string-prompt path is text-only, so images must go here).
        """
        return {
            'type': 'image',
            'source': {
                'type': 'base64',
                'media_type': self.media_type,
                'data': base64.standard_b64encode(self.data).decode('ascii'),
            },
        }


@frozen
class ImageRef:
    """A Slack image attachment located in a message, not yet downloaded.

    ``url`` is the authenticated ``url_private`` (fetching it needs the bot
    token); ``id`` is Slack's file id, used to dedupe the same file appearing
    across replayed messages.
    """

    id: str
    url: str
    media_type: str


def image_files_in(messages: Sequence[dict[str, Any]]) -> list[ImageRef]:
    """Collect supported image attachments across ``messages``, in order.

    Scans each message's ``files`` array, keeping only entries with a vision
    media type (:data:`SUPPORTED_MEDIA_TYPES`) and a fetchable URL. Dedupes by
    file id (first occurrence wins, so the earliest copy in the correspondence
    is the one kept) and caps the result at :data:`MAX_IMAGES`.
    """
    refs: list[ImageRef] = []
    seen: set[str] = set()
    for message in messages:
        for f in message.get('files') or ():
            fid = f.get('id')
            media_type = f.get('mimetype')
            url = f.get('url_private') or f.get('url_private_download')
            if not (fid and url) or media_type not in SUPPORTED_MEDIA_TYPES:
                continue
            if fid in seen:
                continue
            seen.add(fid)
            refs.append(ImageRef(id=fid, url=url, media_type=media_type))
            if len(refs) >= MAX_IMAGES:
                return refs
    return refs


async def download_images(
    refs: Sequence[ImageRef],
    *,
    token: str,
    fetch: Callable[[str, str], Awaitable[bytes]] | None = None,
) -> list[ImageBlob]:
    """Download each ref into an :class:`ImageBlob`, skipping any that fail.

    ``fetch(url, token) -> bytes`` performs the authenticated GET; it defaults
    to an aiohttp client sending ``Authorization: Bearer <token>`` (Slack's
    ``url_private`` requires the bot token). A download that errors or exceeds
    :data:`MAX_IMAGE_BYTES` is logged and skipped — one bad attachment must not
    sink the whole turn — so the model still sees every image that loaded.
    """
    fetch = fetch or _aiohttp_fetch
    blobs: list[ImageBlob] = []
    for ref in refs:
        try:
            data = await fetch(ref.url, token)
        except Exception as exc:  # noqa: BLE001 — degrade per-image, never sink the turn
            _logger.warning('Could not download Slack image %s (%s): %s', ref.id, ref.url, exc)
            continue
        if len(data) > MAX_IMAGE_BYTES:
            _logger.warning(
                'Skipping oversize Slack image %s: %d bytes > %d',
                ref.id, len(data), MAX_IMAGE_BYTES,
            )
            continue
        blobs.append(ImageBlob(media_type=ref.media_type, data=data))
    return blobs


async def _aiohttp_fetch(url: str, token: str) -> bytes:  # pragma: no cover
    """Authenticated GET of a Slack ``url_private`` (the runtime download path).

    Thin adapter over aiohttp (imported lazily — it's only needed when an image
    is actually present), kept out of :func:`download_images` so that function
    stays unit-testable with an injected fetcher.
    """
    import aiohttp

    headers = {'Authorization': f'Bearer {token}'}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as resp:
            resp.raise_for_status()
            return await resp.read()
