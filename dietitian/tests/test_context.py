import asyncio
import copy
import unittest

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

from agents.context import (ContextBudgetExceeded, ContextSettings, ConversationContextMiddleware,
                            estimate_messages, fallback_memory, shape_history, summary_view)
from schemas.chat import ChatRequest


class Summarizer:
    def __init__(self):
        self.calls = []

    async def summarize(self, previous, entries, limit):
        self.calls.append((previous, entries))
        return '用户明确说对花生过敏；保留该禁忌。'


def request(messages, state=None, tools=None):
    return ModelRequest(model=FakeListChatModel(responses=['ok']), messages=messages,
        state=state or {'messages': messages}, tools=tools or [],
        system_message=SystemMessage(content='营养师规则及本轮参考资料'))


class ContextTests(unittest.TestCase):
    def test_old_images_removed_current_images_retained_without_mutating_state(self):
        messages = [HumanMessage(content=[{'type': 'text', 'text': '上次食材'},
                         {'type': 'image', 'url': 'https://example.org/old.jpg'}]),
                    AIMessage(content='收到'),
                    HumanMessage(content=[{'type': 'text', 'text': '这次食材'},
                         {'type': 'image', 'url': 'https://example.org/new.jpg'}])]
        before = copy.deepcopy(messages)
        shaped = shape_history(messages, ContextSettings())
        self.assertNotIn('old.jpg', str(shaped[0].content))
        self.assertIn('上次食材', shaped[0].content)
        self.assertIn('new.jpg', str(shaped[-1].content))
        self.assertEqual(messages, before)

    def test_tools_are_bounded_and_call_result_pairing_is_kept(self):
        messages = [HumanMessage(content='older ' * 6000, id='old'), AIMessage(content='old answer'),
            HumanMessage(content='current question', id='new'),
            AIMessage(content='', tool_calls=[{'name': 'search', 'args': {'q': 'tofu'}, 'id': 'call-1'}]),
            ToolMessage(content='result ' * 6000, tool_call_id='call-1', artifact={'large': 'x' * 1000})]
        prepared = ConversationContextMiddleware(ContextSettings(window_tokens=2000, output_tokens=200)).prepare(request(messages))
        self.assertEqual([m.type for m in prepared.messages], ['human', 'ai', 'tool'])
        self.assertEqual(prepared.messages[1].tool_calls[0]['id'], prepared.messages[2].tool_call_id)
        self.assertLessEqual(len(prepared.messages[-1].content), 4000)
        self.assertEqual(len(messages[-1].content), 42000)
        self.assertIsNotNone(messages[-1].artifact)
        self.assertEqual(prepared.model_settings['max_tokens'], 200)

    def test_summary_view_uses_stable_message_boundary(self):
        messages = [HumanMessage(content='old', id='a'), AIMessage(content='answer', id='b'),
                    HumanMessage(content='new', id='c')]
        memory, tail = summary_view(messages, {'context_summary': 'memory', 'context_summary_until': 'b'})
        self.assertEqual(memory, 'memory')
        self.assertEqual(tail, [messages[-1]])
        self.assertEqual(summary_view(messages, {'context_summary': 'stale', 'context_summary_until': 'missing'}), ('', messages))

    def test_summary_is_saved_separately_and_full_transcript_survives(self):
        summarizer = Summarizer()
        middleware = ConversationContextMiddleware(ContextSettings(summary_messages=4, keep_turns=1), summarizer)
        agent = create_agent(model=FakeListChatModel(responses=['回答']), middleware=[middleware], checkpointer=InMemorySaver())
        async def exercise():
            config = {'configurable': {'thread_id': 'isolated-a'}}
            for question in ['我对花生过敏', '推荐午餐', '还有其他做法吗', '推荐晚餐']:
                await agent.ainvoke({'messages': [HumanMessage(content=question)]}, config)
            state = (await agent.aget_state(config)).values
            self.assertEqual(len(state['messages']), 8)
            self.assertIn('花生过敏', state['context_summary'])
            self.assertTrue(state['context_summary_until'])
            self.assertFalse(any('保留该禁忌' in str(m.content) for m in state['messages']))
            other = await agent.ainvoke({'messages': [HumanMessage(content='hello')]},
                {'configurable': {'thread_id': 'isolated-b'}})
            self.assertNotIn('context_summary', other)
        asyncio.run(exercise())
        self.assertTrue(summarizer.calls)

    def test_summary_failure_retains_checkpoint_and_fallback_quotes_constraints(self):
        class Broken:
            async def summarize(self, *args):
                raise RuntimeError('secret must not leak')
        messages = [HumanMessage(content='我对花生过敏', id='a'), AIMessage(content='收到', id='b'),
                    HumanMessage(content='context ' * 3000, id='c')]
        state = {'messages': messages}
        middleware = ConversationContextMiddleware(ContextSettings(summary_messages=2, keep_turns=1), Broken())
        self.assertIsNone(asyncio.run(middleware.abefore_model(state, None)))
        self.assertNotIn('context_summary_until', state)
        self.assertIn('我对花生过敏', fallback_memory(messages, 2400))

    def test_summary_excludes_tools_and_image_urls_and_batches_all_older_text(self):
        summarizer = Summarizer()
        messages = [HumanMessage(content='早期限制' * 2500, id='a'),
                    AIMessage(content='', tool_calls=[{'name': 'search', 'args': {}, 'id': 't'}], id='b'),
                    ToolMessage(content='UNTRUSTED_TOOL_RESULT', tool_call_id='t', id='c'),
                    HumanMessage(content='new', id='d')]
        middleware = ConversationContextMiddleware(ContextSettings(summary_messages=2, keep_turns=1), summarizer)
        update = asyncio.run(middleware.abefore_model({'messages': messages}, None))
        self.assertEqual(update['context_summary_until'], 'c')
        entries = [entry for _, batch in summarizer.calls for entry in batch]
        self.assertEqual(''.join(entry['text'] for entry in entries), '早期限制' * 2500)
        self.assertNotIn('UNTRUSTED_TOOL_RESULT', str(entries))
        self.assertGreater(len(summarizer.calls), 1)

    def test_rag_system_prompt_and_tool_schemas_count_toward_budget(self):
        middleware = ConversationContextMiddleware(ContextSettings(window_tokens=1000, output_tokens=100))
        model_request = request([HumanMessage(content='hello')])
        model_request = model_request.override(system_message=SystemMessage(content='资料' * 1000))
        with self.assertRaises(ContextBudgetExceeded):
            middleware.prepare(model_request)
        model_request = model_request.override(system_message=SystemMessage(content='rules'),
            tools=[{'name': 'huge', 'description': 'x' * 5000}])
        with self.assertRaises(ContextBudgetExceeded):
            middleware.prepare(model_request)

    def test_oversized_current_user_is_not_silently_truncated(self):
        message = HumanMessage(content='用户条件' * 1000)
        middleware = ConversationContextMiddleware(ContextSettings(window_tokens=1000, output_tokens=100))
        with self.assertRaises(ContextBudgetExceeded):
            middleware.prepare(request([message]))
        self.assertEqual(message.content, '用户条件' * 1000)

    def test_image_token_reserve_is_counted(self):
        message = HumanMessage(content=[{'type': 'image', 'url': 'https://example.org/a.jpg'}])
        self.assertGreaterEqual(estimate_messages([message], image_tokens=2048), 2048)

    def test_chat_input_limits(self):
        self.assertEqual(ChatRequest(message='hello', thread_id='thread-a').image_url, [])
        for fields in ({'message': 'x' * 8001, 'thread_id': 'a'},
                       {'message': 'ok', 'thread_id': ''},
                       {'message': 'ok', 'thread_id': 'a', 'image_url': ['https://example.org/a.jpg'] * 5}):
            with self.assertRaises(ValidationError):
                ChatRequest(**fields)
