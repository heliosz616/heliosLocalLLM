"""Render plaintext local reasoning and its answer as one assistant turn.

The public Responses item history remains intact in ResponseStore. This only
adapts the local backend input, whose current Responses converter otherwise
introduces an extra empty assistant message. Opaque reasoning stays untouched.
Official history contract: https://developers.openai.com/api/docs/guides/conversation-state
"""


def local_reasoning_messages(items):
    result = []
    index = 0
    while index < len(items):
        item = items[index]
        if item.get('type') == 'reasoning' and index + 1 < len(items):
            following = items[index + 1]
            content = item.get('content')
            parts = content if isinstance(content, list) and content else item.get('summary', [])
            text = '\n'.join(part['text'] for part in parts
                             if isinstance(part, dict) and isinstance(part.get('text'), str)) if isinstance(parts, list) else ''
            existing = following.get('reasoning_content', '')
            if (text and following.get('role') == 'assistant' and
                    following.get('type', 'message') == 'message' and isinstance(existing, str)):
                combined = dict(following)
                combined['reasoning_content'] = text + ('\n' + existing if existing else '')
                result.append(combined)
                index += 2
                continue
        result.append(item)
        index += 1
    return result
