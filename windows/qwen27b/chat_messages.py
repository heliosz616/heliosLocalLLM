"""Adapt leading system blocks to the model's single-system native template."""


def normalize_tool_choice(body):
    """Translate a named OpenAI choice into native required + one allowed tool."""
    choice = body.get('tool_choice')
    if not isinstance(choice, dict):
        return
    function = choice.get('function')
    if choice.get('type') != 'function' or not isinstance(function, dict):
        raise ValueError('tool_choice must name a function')
    name = function.get('name')
    tools = body.get('tools')
    if not isinstance(name, str) or not name or not isinstance(tools, list):
        raise ValueError('Named tool_choice requires a declared function')
    selected = [tool for tool in tools if isinstance(tool, dict)
                and tool.get('type') == 'function' and isinstance(tool.get('function'), dict)
                and tool['function'].get('name') == name]
    if len(selected) != 1:
        raise ValueError('Named tool_choice must match exactly one declared function')
    body['tools'] = selected
    body['tool_choice'] = 'required'


def normalize_messages(messages):
    if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
        raise ValueError('messages must be an array of message objects')
    count = 0
    for message in messages:
        if message.get('role') != 'system':
            break
        count += 1
    if any(m.get('role') == 'system' for m in messages[count:]):
        raise ValueError('System messages must precede all conversation turns; this model cannot represent a later system message')
    if count <= 1:
        return messages
    blocks = []
    for message in messages[:count]:
        content = message.get('content')
        if isinstance(content, str):
            blocks.append(content)
        elif isinstance(content, list) and all(isinstance(p, dict) and p.get('type') == 'text' and isinstance(p.get('text'), str) for p in content):
            blocks.append(''.join(p['text'] for p in content))
        else:
            raise ValueError('Leading system messages must contain text')
    return [{'role': 'system', 'content': '\n\n'.join(blocks)}, *messages[count:]]
