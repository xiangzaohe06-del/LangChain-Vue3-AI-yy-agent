"""Tavily search over remote MCP; no direct Tavily SDK/API fallback."""
from __future__ import annotations

import asyncio
from datetime import timedelta
import json
import logging
import os
from urllib.parse import urlsplit

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

log = logging.getLogger(__name__)


class TavilyMCP:
    def __init__(self, client_factory=None):
        self._factory = client_factory
        self._lock = asyncio.Lock()
        self._tool = None
        self._signature = None

    @staticmethod
    def configuration():
        url = os.getenv('TAVILY_MCP_URL', 'https://mcp.tavily.com/mcp/').strip()
        parsed = urlsplit(url)
        if (not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.scheme not in {'http', 'https'}):
            raise ValueError('Invalid MCP endpoint; credentials belong in headers')
        if parsed.scheme == 'http' and parsed.hostname not in {'localhost', '127.0.0.1', '::1'}:
            raise ValueError('Remote MCP requires HTTPS')
        key = (os.getenv('TAVILY_API_KEY') or os.getenv('TVILY_API_KEY') or '').strip()
        if not key:
            raise ValueError('Missing Tavily API key')
        timeout = max(5, min(120, int(os.getenv('TAVILY_MCP_TIMEOUT', '30'))))
        connection = {
            'url': url, 'transport': 'streamable_http',
            'headers': {'Authorization': 'Bearer ' + key,
                        'DEFAULT_PARAMETERS': json.dumps({'max_results': 5, 'topic': 'general',
                            'include_images': True, 'include_raw_content': False, 'search_depth': 'basic'})},
            'timeout': timedelta(seconds=timeout), 'sse_read_timeout': timedelta(seconds=timeout),
        }
        return connection, timeout, (url, key, timeout)

    async def _discover(self, connection, signature):
        async with self._lock:
            if self._tool is not None and self._signature == signature:
                return self._tool
            factory = self._factory
            if factory is None:
                from langchain_mcp_adapters.client import MultiServerMCPClient
                factory = MultiServerMCPClient
            client = factory({'tavily': connection}, handle_tool_errors=False)
            tools = await client.get_tools(server_name='tavily')
            matches = [candidate for candidate in tools
                       if candidate.name.replace('-', '_') == 'tavily_search']
            if len(matches) != 1 or 'query' not in matches[0].args:
                raise ValueError('Tavily MCP search tool missing or ambiguous')
            self._tool, self._signature = matches[0], signature
            return self._tool

    @staticmethod
    def normalize(value):
        if isinstance(value, ToolMessage):
            if value.status == 'error':
                raise ValueError('Remote tool failed')
            value = value.content
        if isinstance(value, list):
            value = '\n'.join(block.get('text', '') for block in value
                              if isinstance(block, dict) and block.get('type') == 'text')
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                if not value.strip():
                    raise ValueError('Empty MCP search response')
                return {'backend': 'tavily_mcp', 'content': value[:12000], 'truncated': len(value) > 12000}
        if not isinstance(value, dict) or value.get('error'):
            raise ValueError('Invalid MCP search response')
        results = value.get('results')
        if not isinstance(results, list):
            raise ValueError('Missing search results')
        projected = []
        for result in results[:5]:
            if not isinstance(result, dict):
                raise ValueError('Invalid search result')
            projected.append({'title': str(result.get('title', ''))[:300],
                              'url': str(result.get('url', ''))[:2000],
                              'content': str(result.get('content', ''))[:2000]})
        images = value.get('images', [])
        return {'backend': 'tavily_mcp', 'results': projected,
                'images': images[:5] if isinstance(images, list) else []}

    async def search(self, query):
        try:
            if not isinstance(query, str) or not query.strip() or len(query) > 1000:
                raise ValueError('Search query must contain 1 to 1000 characters')
            connection, timeout, signature = self.configuration()
            async with asyncio.timeout(timeout):
                remote = await self._discover(connection, signature)
                arguments = {'query': query.strip(), 'max_results': 5, 'topic': 'general',
                             'include_images': True, 'include_raw_content': False, 'search_depth': 'basic'}
                arguments = {key: value for key, value in arguments.items() if key in remote.args}
                result = self.normalize(await remote.ainvoke(arguments))
            # A misbehaving server must not echo our credential into chat/history.
            encoded = json.dumps(result, ensure_ascii=False).replace(signature[1], '[redacted]')
            return json.loads(encoded)
        except Exception as error:
            self._tool = self._signature = None
            log.warning('Tavily MCP search unavailable (%s)', type(error).__name__)
            return {'backend': 'tavily_mcp', 'success': False, 'results': [],
                    'error': '联网搜索暂不可用，请检查Tavily密钥、MCP地址或网络；本次没有可引用的联网搜索结果。'}


tavily_mcp = TavilyMCP()


@tool
async def web_search(query: str) -> str:
    """通过Tavily MCP联网搜索菜谱及公开资料，返回标题、来源链接、摘要和图片。query最多1000字符；失败时不可声称已联网检索。"""
    return json.dumps(await tavily_mcp.search(query), ensure_ascii=False)
