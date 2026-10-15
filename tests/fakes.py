'''Builders for test doubles shared across test files.'''

from agentic_job_search.triage import ChatReply

FAKE_CHAT_MODEL = 'fake/model'
FAKE_CHAT_PROVIDER = 'FakeProvider'


def chat_reply(content: str, cost_usd: float = 0.0, model: str = FAKE_CHAT_MODEL) -> ChatReply:
    '''What a stubbed `chat_openrouter` returns: `content`, served by a fake provider.'''
    return ChatReply(content=content, cost_usd=cost_usd, requested_model=model, model=model,
                     provider=FAKE_CHAT_PROVIDER, served_quantization='fp8', finish_reason='stop',
                     usage={'prompt_tokens': 1, 'completion_tokens': 1})
