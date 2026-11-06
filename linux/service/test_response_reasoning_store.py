"""Keep durable Responses items intact while rendering one local assistant turn."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gateway import ProtocolError, ResponseStore


def test_reasoning_chain_rendering_preserves_durable_items_and_references(tmp_path):
    path = tmp_path / 'responses.sqlite3'
    store = ResponseStore(path)
    inputs = [{'role': 'system', 'content': 'first'},
              {'role': 'system', 'content': 'second'},
              {'role': 'user', 'content': 'question'}]
    reasoning = {'id': 'rs_1', 'type': 'reasoning',
                 'summary': [{'type': 'summary_text', 'text': 'analysis'}]}
    answer = {'id': 'msg_1', 'type': 'message', 'role': 'assistant',
              'content': [{'type': 'output_text', 'text': 'answer', 'annotations': []}]}
    response = {'id': 'resp_1', 'output': [reasoning, answer]}
    original = copy.deepcopy(response)
    store.save('owner', 'first-model', inputs, response)
    store.close()
    store = ResponseStore(path)
    payload = {'previous_response_id': 'resp_1',
               'input': [{'role': 'user', 'content': 'follow-up'}]}
    body, durable = store.prepare(payload, 'owner', 'second-model')
    assert durable == inputs + [reasoning, answer] + payload['input']
    assert body['input'] == inputs + [dict(answer, reasoning_content='analysis')] + payload['input']
    assert store.get('resp_1', 'owner')[2] == original
    store.save('owner', 'second-model', durable, {'id': 'resp_2', 'output': []})
    assert store.get('resp_2', 'owner')[1] == durable
    references = [{'type': 'item_reference', 'id': 'rs_1'},
                  {'type': 'item_reference', 'id': 'msg_1'}]
    rendered, expanded = store.prepare({'input': references}, 'owner', 'second-model')
    assert expanded == [reasoning, answer]
    assert rendered['input'] == [dict(answer, reasoning_content='analysis')]
    with pytest.raises(ProtocolError):
        store.prepare(payload, 'another-owner', 'second-model')
    store.close()


def test_tool_roundtrip_keeps_reasoning_call_order_and_original_call_ids(tmp_path):
    store = ResponseStore(tmp_path / 'responses.sqlite3')
    items = [{'id': 'rs_t', 'type': 'reasoning',
              'summary': [{'type': 'summary_text', 'text': 'use tools'}]},
             {'id': 'fc_1', 'type': 'function_call', 'call_id': 'call_1',
              'name': 'lookup', 'arguments': '{"key":"one"}'},
             {'id': 'fc_2', 'type': 'function_call', 'call_id': 'call_2',
              'name': 'lookup', 'arguments': '{"key":"two"}'}]
    store.save('owner', 'model', [], {'id': 'resp_tools', 'output': items})
    outputs = [{'type': 'function_call_output', 'call_id': 'call_2', 'output': 'two'},
               {'type': 'function_call_output', 'call_id': 'call_1', 'output': 'one'}]
    body, durable = store.prepare({'previous_response_id': 'resp_tools', 'input': outputs}, 'owner', 'model')
    assert body['input'] == durable == items + outputs
    store.close()
