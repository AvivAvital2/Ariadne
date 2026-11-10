"""Offline guardrails for exact production short-call diagnostic captures."""

import hashlib
import json
import socket

import pytest


async def test_capture_uses_production_paths_and_preserves_each_budget(tmp_path, monkeypatch):
    from tests.capture_mac_short_calls import capture

    def forbid_network(*_args, **_kwargs):
        raise AssertionError('Offline capture must not connect to any model')

    monkeypatch.setattr(socket.socket, 'connect', forbid_network)
    output = tmp_path / 'capture'
    protocol = await capture(output)
    cases = {case['name']: case for case in protocol['cases']}
    assert {name: case['native_request']['max_tokens'] for name, case in cases.items()} == {
        'compact-plan': 2432, 'compact-family': 2112, 'compact-answer': 4096,
        'cascade-component': 2304, 'cascade-route': 2304, 'cascade-body': 2176,
        'cascade-answer': 4096, 'catalog': 2048, 'theme': 2048, 'trace': 1024,
    }
    for name, case in cases.items():
        native, hosted = case['native_request'], case['hosted_request']
        assert native['messages'] == hosted['messages']
        assert native['max_tokens'] == hosted['max_tokens']
        assert native['reasoning_effort'] == hosted['reasoning_effort'] == 'xhigh'
        assert native['chat_template_kwargs'] == hosted['chat_template_kwargs'] == {'enable_thinking': True}
        assert native['reasoning_format'] == 'deepseek'
        assert 'reasoning_format' not in hosted and 'repeat_penalty' not in hosted
        assert hosted['repetition_penalty'] == native['repeat_penalty'] == 1
        assert hosted['model'] == 'Qwen/Qwen3.8-27B'
        assert case['system_prompt'] == native['messages'][0]['content']
        assert case['user_prompt'] == native['messages'][1]['content']
        assert case['max_tokens'] == native['max_tokens']
        assert case['timeout_seconds'] == (30 if name == 'trace' else 600)
        assert 'expected' not in hosted
    assert 'pkg.Writer.write' in cases['compact-family']['hosted_request']['messages'][1]['content']
    assert 'Widget.flush' in cases['cascade-body']['hosted_request']['messages'][1]['content']
    assert cases['catalog']['native_request']['messages'][0]['content'] == ''
    assert cases['trace']['native_request']['messages'][0]['content'] == ''
    assert 'Cursor: w.Widget.run' in cases['trace']['native_request']['messages'][1]['content']
    assert 'Reply with' in cases['compact-plan']['native_request']['messages'][0]['content']
    assert protocol['network_calls'] == 0
    assert protocol['end_to_end_quality_qualified'] is False
    assert 'scripted' in protocol['downstream_input_limit']
    for path, digest in protocol['source_hashes'].items():
        assert hashlib.sha256((output / 'provenance' / path).read_bytes()).hexdigest() == digest
    assert json.loads((output / 'protocol.json').read_text()) == protocol


async def test_capture_refuses_to_overwrite_frozen_inputs(tmp_path):
    from tests.capture_mac_short_calls import capture

    output = tmp_path / 'capture'
    output.mkdir()
    sentinel = output / 'protocol.json'
    sentinel.write_text('immutable')
    with pytest.raises(FileExistsError):
        await capture(output)
    assert sentinel.read_text() == 'immutable'
