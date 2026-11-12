'''Builders for test doubles shared across test files.'''

import json

from agentic_job_search.triage import ChatReply

FAKE_CHAT_MODEL = 'fake/model'
FAKE_CHAT_PROVIDER = 'FakeProvider'


def chat_reply(content: str, cost_usd: float = 0.0, model: str = FAKE_CHAT_MODEL) -> ChatReply:
    '''What a stubbed `chat_openrouter` returns: `content`, served by a fake provider.'''
    return ChatReply(content=content, cost_usd=cost_usd, requested_model=model, model=model,
                     provider=FAKE_CHAT_PROVIDER, served_quantization='fp8', finish_reason='stop',
                     usage={'prompt_tokens': 1, 'completion_tokens': 1})


def openrouter_tool_reply(tool_name: str, arguments: str, cost_usd: float = 0.0, model: str = FAKE_CHAT_MODEL,
                          finish_reason: str = 'tool_calls') -> str:
    '''What the OpenRouter tool server returns for a forced function call whose arguments the model wrote
    as `arguments` (a JSON string, possibly broken): for stubbing triage.call_mcp_tool.'''
    return json.dumps({'ok': True, 'content': None, 'cost_usd': cost_usd, 'model': model, 'provider': FAKE_CHAT_PROVIDER,
                       'served_quantization': 'fp8', 'finish_reason': finish_reason,
                       'usage': {'prompt_tokens': 1, 'completion_tokens': 1},
                       'tool_calls': [{'id': 'c1', 'type': 'function',
                                       'function': {'name': tool_name, 'arguments': arguments}}]})
