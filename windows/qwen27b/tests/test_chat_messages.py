import pytest
from chat_messages import normalize_messages, normalize_tool_choice


def test_named_tool_choice_enforces_only_declared_function():
    tools = [{'type':'function','function':{'name':name}} for name in ('first','second')]
    body = {'tools':tools,'tool_choice':{'type':'function','function':{'name':'second'}}}
    normalize_tool_choice(body)
    assert body['tools'] == [tools[1]]
    assert body['tool_choice'] == 'required'
    assert len(tools) == 2


def test_unknown_named_tool_is_rejected_instead_of_becoming_auto():
    body = {'tools':[],'tool_choice':{'type':'function','function':{'name':'missing'}}}
    with pytest.raises(ValueError):
        normalize_tool_choice(body)


def test_leading_systems_preserve_order_and_tool_turns():
    messages = [{'role':'system','content':'policy A'},
        {'role':'system','content':[{'type':'text','text':'policy B'}]},
        {'role':'user','content':'question'},
        {'role':'assistant','tool_calls':[{'id':'one'}]},
        {'role':'tool','tool_call_id':'one','content':'result'}]
    result = normalize_messages(messages)
    assert result[0] == {'role':'system','content':'policy A\n\npolicy B'}
    assert result[1:] == messages[2:]
    assert len(messages) == 5


def test_single_system_is_unchanged():
    messages = [{'role':'system','content':'policy'}, {'role':'user','content':'q'}]
    assert normalize_messages(messages) is messages


@pytest.mark.parametrize('messages', [None, ['bad'],
    [{'role':'user','content':'q'},{'role':'system','content':'late'}],
    [{'role':'system','content':'a'},{'role':'system','content':[{'type':'image_url'}]}]])
def test_unrepresentable_messages_fail_explicitly(messages):
    with pytest.raises(ValueError):
        normalize_messages(messages)
