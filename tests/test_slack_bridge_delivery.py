from __future__ import annotations

import pytest

from slack_bridge.delivery import (
    FALLBACK_CHUNK_MAX_CHARS,
    markdown_attachment_requested,
    split_slack_message,
)


@pytest.mark.parametrize(
    'request_text',
    [
        'Attach your complete answer as a Markdown file.',
        'Return the full plan as an attached .md file.',
        'Please upload the response as Markdown.',
        'Put the elaborate prompt into a file attachment.',
        'Respond with a Markdown document.',
        '--attach-markdown Produce the migration plan.',
        'Attach a markdown with plans that require a lot of text.',
        'Give me the answer as a file.',
        'Create a Markdown file with the plan.',
    ],
)
def test_markdown_attachment_requested_accepts_explicit_output_requests(request_text):
    assert markdown_attachment_requested(request_text)


@pytest.mark.parametrize(
    'request_text',
    [
        'Produce the complete implementation plan.',
        'How does Markdown file attachment work?',
        'Find the code that uploads a file to Slack.',
        'Explain README.md.',
        'The plan mentions file attachments.',
        'Do not attach a file; reply in chat.',
        'Answer without a Markdown attachment.',
    ],
)
def test_markdown_attachment_requested_rejects_mentions_and_negation(request_text):
    assert not markdown_attachment_requested(request_text)


def test_split_slack_message_is_lossless_and_bounded_for_hard_wrapped_text():
    text = 'x' * (FALLBACK_CHUNK_MAX_CHARS * 2 + 17)

    chunks = split_slack_message(text)

    assert ''.join(chunks) == text
    assert all(0 < len(chunk) <= FALLBACK_CHUNK_MAX_CHARS for chunk in chunks)


def test_split_slack_message_prefers_complete_paragraph_boundaries():
    paragraph = 'one paragraph\n\n'
    text = paragraph * 20

    chunks = split_slack_message(text, max_chars=len(paragraph) * 3 + 4)

    assert ''.join(chunks) == text
    assert all(chunk.endswith('\n\n') for chunk in chunks[:-1])


def test_split_slack_message_handles_empty_text_and_rejects_invalid_limit():
    assert split_slack_message('') == ['']
    with pytest.raises(ValueError, match='positive'):
        split_slack_message('answer', max_chars=0)


def test_split_slack_message_keeps_boundary_separator_within_limit():
    text = 'x' * FALLBACK_CHUNK_MAX_CHARS + '\n\nrest'

    chunks = split_slack_message(text)

    assert ''.join(chunks) == text
    assert all(len(chunk) <= FALLBACK_CHUNK_MAX_CHARS for chunk in chunks)
