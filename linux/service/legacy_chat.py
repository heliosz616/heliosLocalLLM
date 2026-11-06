"""Translate the deprecated OpenAI function schema at the protocol boundary."""
import hashlib
import json


def prepare(body):
    messages = body.get('messages')
    legacy = 'functions' in body or 'function_call' in body or (
        isinstance(messages, list) and any(isinstance(m, dict) and
            (m.get('role') == 'function' or 'function_call' in m) for m in messages))
    if not legacy:
        return body, False
    result = dict(body)
    functions = result.pop('functions', None)
    if functions is not None and 'tools' not in result:
        if not isinstance(functions, list):
            raise ValueError('functions must be an array')
        result['tools'] = [{'type': 'function', 'function': f} for f in functions]
    choice = result.pop('function_call', None)
    if choice is not None and 'tool_choice' not in result:
        result['tool_choice'] = ({'type': 'function', 'function': choice}
                                 if isinstance(choice, dict) else choice)
    if isinstance(messages, list):
        converted, calls = [], []
        for index, original in enumerate(messages):
            if not isinstance(original, dict):
                converted.append(original)
                continue
            message = dict(original)
            call = message.pop('function_call', None)
            if call is not None:
                if not isinstance(call, dict):
                    raise ValueError('message.function_call must be an object')
                encoded = json.dumps([index, call], sort_keys=True).encode()
                cid = 'call_legacy_' + hashlib.sha256(encoded).hexdigest()[:20]
                message['tool_calls'] = [{'id': cid, 'type': 'function', 'function': call}]
                calls.append((call.get('name'), cid))
            if message.get('role') == 'function':
                # A function result follows the latest unresolved call of that
                # name. IDs depend on the supplied history, never on a session.
                name = message.get('name')
                match = next((i for i in range(len(calls) - 1, -1, -1)
                              if calls[i][0] == name), None)
                if match is not None:
                    _, message['tool_call_id'] = calls.pop(match)
                message['role'] = 'tool'
                message.pop('name', None)
            converted.append(message)
        result['messages'] = converted
    # Legacy function_call represents a single call, unlike modern tool_calls.
    result.setdefault('parallel_tool_calls', False)
    return result, True


def response(value, called=None):
    if called is None:
        called = set()
    for choice in value.get('choices', []):
        for field in ('message', 'delta'):
            message = choice.get(field)
            if not isinstance(message, dict):
                continue
            calls = message.pop('tool_calls', None)
            if calls:
                called.add(choice.get('index', 0))
                function = calls[0].get('function')
                if function is not None:
                    message['function_call'] = function
        if choice.get('finish_reason') == 'tool_calls' or (
                choice.get('index', 0) in called and choice.get('finish_reason') == 'stop'):
            choice['finish_reason'] = 'function_call'
    return value


async def stream(chunks):
    pending = b''
    called = set()
    async for chunk in chunks:
        pending = (pending + chunk).replace(b'\r\n', b'\n')
        while b'\n\n' in pending:
            event, pending = pending.split(b'\n\n', 1)
            lines = []
            for line in event.split(b'\n'):
                if line.startswith(b'data:') and line[5:].strip() != b'[DONE]':
                    value = response(json.loads(line[5:]), called)
                    line = b'data: ' + json.dumps(value, ensure_ascii=False).encode()
                lines.append(line)
            yield b'\n'.join(lines) + b'\n\n'
    if pending:
        yield pending
