import asyncio
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from langchain_core.messages import ToolMessage
from agents.tavily_mcp import TavilyMCP, web_search


class FakeTool:
    name = 'tavily-search'
    args = {key: {} for key in ('query', 'max_results', 'topic', 'include_images', 'include_raw_content', 'search_depth')}

    def __init__(self):
        self.calls = []
        self.result = [{'type': 'text', 'text': json.dumps({
            'results': [{'title': 'Recipe', 'url': 'https://example.org/recipe', 'content': 'tofu recipe'}],
            'images': ['https://example.org/image.jpg']})}]
        self.error = None

    async def ainvoke(self, arguments):
        self.calls.append(arguments)
        if self.error:
            raise self.error
        return self.result


class Factory:
    def __init__(self, tools):
        self.tools = tools
        self.connections = []
        self.discoveries = 0

    def __call__(self, connections, **kwargs):
        self.connections.append((connections, kwargs))
        async def get_tools(**kwargs):
            self.discoveries += 1
            await asyncio.sleep(0)
            return self.tools
        return SimpleNamespace(get_tools=get_tools)


class TavilyMCPTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'TAVILY_API_KEY': 'tvly-unit-secret', 'TVILY_API_KEY': '',
            'TAVILY_MCP_URL': 'https://mcp.tavily.com/mcp/', 'TAVILY_MCP_TIMEOUT': '30'})
        env.start()
        self.addCleanup(env.stop)
        self.remote = FakeTool()
        self.factory = Factory([SimpleNamespace(name='tavily-crawl'), self.remote])
        self.service = TavilyMCP(self.factory)

    async def test_remote_discovery_authentication_and_fixed_search_arguments(self):
        result = await self.service.search('tofu recipe')
        connection = self.factory.connections[0][0]['tavily']
        self.assertEqual(connection['transport'], 'streamable_http')
        self.assertEqual(connection['headers']['Authorization'], 'Bearer tvly-unit-secret')
        self.assertNotIn('tvly-unit-secret', connection['url'])
        self.assertFalse(self.factory.connections[0][1]['handle_tool_errors'])
        self.assertEqual(self.remote.calls[0]['max_results'], 5)
        self.assertEqual(self.remote.calls[0]['topic'], 'general')
        self.assertTrue(self.remote.calls[0]['include_images'])
        self.assertEqual(result['backend'], 'tavily_mcp')
        self.assertEqual(result['results'][0]['url'], 'https://example.org/recipe')

    async def test_discovery_cached_and_concurrent_calls_do_not_double_discover(self):
        results = await asyncio.gather(self.service.search('a'), self.service.search('b'))
        self.assertEqual(self.factory.discoveries, 1)
        self.assertEqual(len(self.remote.calls), 2)
        self.assertTrue(all(result['results'] for result in results))

    async def test_only_remote_supported_parameters_are_sent(self):
        self.remote.args = {'query': {}}
        await self.service.search('a')
        self.assertEqual(self.remote.calls, [{'query': 'a'}])

    async def test_canonical_and_legacy_key_and_configuration_change(self):
        await self.service.search('a')
        with patch.dict(os.environ, {'TAVILY_API_KEY': '', 'TVILY_API_KEY': 'tvly-legacy-secret'}):
            await self.service.search('b')
        self.assertEqual(self.factory.discoveries, 2)
        connection = self.factory.connections[-1][0]['tavily']
        self.assertEqual(connection['headers']['Authorization'], 'Bearer tvly-legacy-secret')

    async def test_missing_key_or_invalid_query_does_not_connect(self):
        with patch.dict(os.environ, {'TAVILY_API_KEY': '', 'TVILY_API_KEY': ''}):
            self.assertFalse((await self.service.search('a'))['success'])
        for query in ('', ' ' * 3, 'x' * 1001):
            self.assertFalse((await self.service.search(query))['success'])
        self.assertFalse(self.factory.connections)

    async def test_invalid_endpoint_never_sends_credentials(self):
        for url in ('https://example.org/mcp?key=secret', 'http://example.org/mcp', 'file:///tmp/a',
                    'https://user:secret@example.org/mcp'):
            with patch.dict(os.environ, {'TAVILY_MCP_URL': url}):
                self.assertFalse((await self.service.search('a'))['success'])
        self.assertFalse(self.factory.connections)

    async def test_remote_error_is_sanitized_and_next_call_rediscovers(self):
        self.remote.error = RuntimeError('tvly-unit-secret sensitive transport details')
        result = await self.service.search('a')
        self.assertFalse(result['success'])
        self.assertNotIn('tvly-unit-secret', json.dumps(result))
        self.remote.error = None
        self.assertTrue((await self.service.search('b'))['results'])
        self.assertEqual(self.factory.discoveries, 2)

    async def test_missing_or_ambiguous_search_tool_is_not_replaced_by_crawl(self):
        for tools in ([SimpleNamespace(name='tavily-crawl')], [self.remote, self.remote]):
            service = TavilyMCP(Factory(tools))
            self.assertFalse((await service.search('a'))['success'])

    async def test_error_tool_message_is_not_treated_as_sources(self):
        self.remote.result = ToolMessage(content='secret server details', tool_call_id='a', status='error')
        result = await self.service.search('a')
        self.assertFalse(result['success'])
        self.assertNotIn('secret server details', json.dumps(result))

    async def test_result_count_body_size_and_credential_echo_are_bounded(self):
        self.remote.result = json.dumps({'results': [{'title': 'tvly-unit-secret', 'content': 'x'*5000}] * 8})
        result = await self.service.search('a')
        self.assertEqual(len(result['results']), 5)
        self.assertEqual(len(result['results'][0]['content']), 2000)
        self.assertNotIn('tvly-unit-secret', json.dumps(result))

    async def test_public_web_search_tool_uses_mcp_service(self):
        with patch('agents.tavily_mcp.tavily_mcp', self.service):
            result = json.loads(await web_search.ainvoke({'query': 'tofu'}))
        self.assertEqual(result['backend'], 'tavily_mcp')
        self.assertEqual(self.remote.calls[0]['query'], 'tofu')
