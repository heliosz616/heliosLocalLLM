import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import legacy_chat


def test_round_trip_repeated_names_and_modern_passthrough():
    modern = {'messages': [{'role': 'user', 'content': 'a'}], 'tools': []}
    assert legacy_chat.prepare(modern) == (modern, False)
    body = {'functions': [{'name': 'increment', 'parameters': {}}], 'function_call': {'name': 'increment'},
            'messages': [{'role': 'assistant', 'function_call': {'name': 'increment', 'arguments': '{"value":17}'}},
                         {'role': 'function', 'name': 'increment', 'content': '18'},
                         {'role': 'assistant', 'function_call': {'name': 'increment', 'arguments': '{"value":23}'}},
                         {'role': 'function', 'name': 'increment', 'content': '24'}]}
    value, legacy = legacy_chat.prepare(body)
    assert legacy and 'functions' not in value and 'function_call' not in value
    assert value['tool_choice']['function']['name'] == 'increment'
    messages = value['messages']
    assert messages[0]['tool_calls'][0]['id'] == messages[1]['tool_call_id']
    assert messages[2]['tool_calls'][0]['id'] == messages[3]['tool_call_id']
    assert messages[1]['tool_call_id'] != messages[3]['tool_call_id']
    assert messages[1]['role'] == 'tool' and body['messages'][1]['role'] == 'function'


@pytest.mark.asyncio
async def test_fragmented_stream_and_finish_reason():
    values = [{'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'name': 'increment', 'arguments': ''}}]}, 'finish_reason': None}]},
              {'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': '{"value":17}'}}]}, 'finish_reason': None}]},
              {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}]
    raw = ''.join('data: ' + json.dumps(v) + '\r\n\r\n' for v in values).encode() + b'data: [DONE]\r\n\r\n'
    async def pieces():
        for i in range(0, len(raw), 7):
            yield raw[i:i+7]
    result = b''.join([chunk async for chunk in legacy_chat.stream(pieces())])
    events = [json.loads(event[6:]) for event in result.split(b'\n\n') if event.startswith(b'data: ') and event[6:] != b'[DONE]']
    assert events[0]['choices'][0]['delta']['function_call']['name'] == 'increment'
    assert json.loads(events[1]['choices'][0]['delta']['function_call']['arguments']) == {'value': 17}
    assert events[2]['choices'][0]['finish_reason'] == 'function_call'
