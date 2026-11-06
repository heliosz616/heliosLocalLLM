"""Local rendering regression cases; original protocol items must remain unchanged."""
import copy
import unittest

from reasoning_history import local_reasoning_messages


class ReasoningHistoryTests(unittest.TestCase):
    def reasoning(self, text='analysis'):
        return dict(id='rs_1', type='reasoning', summary=[dict(type='summary_text', text=text)])

    def message(self, text='answer'):
        return dict(id='msg_1', type='message', role='assistant', status='completed',
                    content=[dict(type='output_text', text=text, annotations=[])])

    def test_reasoning_and_answer_render_one_turn_without_mutating_protocol_items(self):
        items = [self.reasoning(), self.message(), dict(role='user', content='next')]
        original = copy.deepcopy(items)
        actual = local_reasoning_messages(items)
        self.assertEqual(items, original)
        self.assertEqual(actual, [dict(items[1], reasoning_content='analysis'), items[2]])
        self.assertEqual(actual[0]['id'], 'msg_1')

    def test_full_content_precedes_summary_and_preserves_text_part_order(self):
        reason = self.reasoning('short summary')
        reason['content'] = [dict(type='reasoning_text', text='first'), dict(type='reasoning_text', text='second')]
        actual = local_reasoning_messages([reason, self.message()])
        self.assertEqual(actual[0]['reasoning_content'], 'first\nsecond')

    def test_genuine_consecutive_assistant_and_repeated_system_messages_keep_order(self):
        items = [dict(role='system', content='one'), self.message('first'), self.message('second'),
                 dict(role='system', content='two'), dict(role='user', content='third')]
        self.assertEqual(local_reasoning_messages(items), items)

    def test_tool_reasoning_and_parallel_calls_remain_for_native_call_ID_conversion(self):
        items = [self.reasoning(), dict(type='function_call', call_id='a', name='f', arguments='{}'),
                 dict(type='function_call', call_id='b', name='f', arguments='{}'),
                 dict(type='function_call_output', call_id='b', output='second'),
                 dict(type='function_call_output', call_id='a', output='first')]
        self.assertEqual(local_reasoning_messages(items), items)

    def test_opaque_or_empty_reasoning_is_never_discarded(self):
        for reason in [dict(type='reasoning', encrypted_content='opaque', summary=[]),
                       dict(type='reasoning', summary=[]), dict(type='reasoning', summary='invalid')]:
            items = [reason, self.message()]
            self.assertEqual(local_reasoning_messages(items), items)

    def test_multiple_turns_and_intervening_roles_are_not_reordered(self):
        items = [self.reasoning('one'), self.message('first'), dict(role='user', content='next'),
                 self.reasoning('two'), self.message('second')]
        actual = local_reasoning_messages(items)
        self.assertEqual([item.get('role') for item in actual], ['assistant', 'user', 'assistant'])
        self.assertEqual([actual[0]['reasoning_content'], actual[2]['reasoning_content']], ['one', 'two'])
        separated = [self.reasoning(), dict(role='system', content='new'), self.message()]
        self.assertEqual(local_reasoning_messages(separated), separated)

    def test_explicit_message_fields_and_additional_plaintext_are_preserved(self):
        message = dict(self.message(), reasoning_content='additional', phase='final_answer', metadata={'keep': 1})
        actual = local_reasoning_messages([self.reasoning(), message])[0]
        self.assertEqual(actual, dict(message, reasoning_content='analysis\nadditional'))


if __name__ == '__main__':
    unittest.main()
