'''Shadow Local Model: the remote answer is what callers get; the local one is only recorded.'''

import asyncio
import json

import pytest
from agentic_job_search import config, shadow, triage

LOCAL_MODEL = 'mlx-lm-server:org/local-model'


@pytest.fixture
def shadow_on(monkeypatch, tmp_path):
    monkeypatch.setattr(shadow, 'SHADOW_CALLS_DIR', tmp_path)
    monkeypatch.setattr(config, 'SHADOW_MODEL', None)
    path = shadow.activate(LOCAL_MODEL, 'test-run')
    yield path
    monkeypatch.setattr(config, 'SHADOW_MODEL', None)


def _servers(monkeypatch, remote_content: str | None, local: str | Exception, local_gate: asyncio.Event | None = None,
             remote_tool_calls: list | None = None):
    '''Fake OpenRouter and local servers behind triage.call_mcp_tool; returns the calls made.'''
    calls: list[tuple[str, dict]] = []

    async def fake_call(url, tool, args, timeout_seconds=None):
        calls.append((tool, args))
        if url == config.LLM_OPENROUTER_MCP_URL:
            return json.dumps({'ok': True, 'content': remote_content, 'tool_calls': remote_tool_calls,
                               'cost_usd': 0.001, 'model': 'vendor/remote', 'provider': 'P',
                               'finish_reason': 'tool_calls' if remote_tool_calls else 'stop', 'usage': {}})
        if local_gate is not None:
            await local_gate.wait()
        if isinstance(local, Exception):
            raise local
        return json.dumps({'response': local, 'done': True, 'done_reason': 'stop'})
    monkeypatch.setattr(triage, 'call_mcp_tool', fake_call)
    return calls


SETTLE_TICKS = 50


async def _settle() -> None:
    '''Let the shadow worker answer everything queued (the fakes answer at once).'''
    for _ in range(SETTLE_TICKS):
        if shadow._in_flight is None and (shadow._queue is None or shadow._queue.empty()):
            return
        await asyncio.sleep(0)
    raise AssertionError('the shadow worker did not settle')


def _records(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


@pytest.mark.parametrize(('content', 'answer', 'level'), [
    ('{"a": 1}\nCONFIDENCE: 4', '{"a": 1}', 4),
    ('{"a": 1}\n\n**Confidence:** 5\n', '{"a": 1}', 5),
    ('{"note": "confidence matters"}', '{"note": "confidence matters"}', None),
    ('CONFIDENCE: 2\n{"a": 1}\nCONFIDENCE: 3', 'CONFIDENCE: 2\n{"a": 1}', 3),
])
def test_split_confidence_cuts_only_the_last_confidence_line(content, answer, level):
    assert shadow.split_confidence(content) == (answer, level)


async def test_the_caller_gets_the_remote_answer_without_its_confidence_line(monkeypatch, shadow_on):
    calls = _servers(monkeypatch, '{"rating": 4}\nCONFIDENCE: 5', '{"rating": 3}\nCONFIDENCE: 2')
    reply = await triage.chat_openrouter('Rate this.', system='sys', model='vendor/remote', temperature=0.2)
    assert reply.content == '{"rating": 4}' and reply.confidence == 5
    remote_prompt = calls[0][1]['messages'][-1]['content']
    assert remote_prompt == 'Rate this.' + shadow.CONFIDENCE_INSTRUCTION
    await _settle()
    summary = await shadow.finish()
    assert summary[shadow.STATUS_PAIRED] == 1
    [record] = _records(shadow_on)
    assert record['status'] == shadow.STATUS_PAIRED
    assert record['prompt'] == remote_prompt and record['system'] == 'sys' and record['temperature'] == 0.2
    assert record['remote']['content'] == '{"rating": 4}' and record['remote']['confidence'] == 5
    assert record['shadow'] == {'model': LOCAL_MODEL, 'content': '{"rating": 3}', 'confidence': 2,
                                'seconds': record['shadow']['seconds']}
    local_args = calls[1][1]
    assert local_args['prompt'] == remote_prompt and local_args['temperature'] == 0.2
    assert local_args['purpose_slug'] == config.LLM_LOCAL_PURPOSE_SHADOW


async def test_a_failing_shadow_is_recorded_and_never_reaches_the_caller(monkeypatch, shadow_on):
    _servers(monkeypatch, '{"ok": true}', RuntimeError('sidecar down'))
    reply = await triage.chat_openrouter('Q?', model='vendor/remote')
    assert reply.content == '{"ok": true}'
    await _settle()
    await shadow.finish()
    [record] = _records(shadow_on)
    assert record['status'] == shadow.STATUS_SHADOW_ERROR and 'sidecar down' in record['shadow']['error']


async def test_calls_unanswered_at_the_end_are_written_as_pending(monkeypatch, shadow_on):
    gate = asyncio.Event()  # never set: the local model never answers
    _servers(monkeypatch, 'yes\nCONFIDENCE: 4', 'unused', local_gate=gate)
    await triage.chat_openrouter('first', model='vendor/remote')
    await triage.chat_openrouter('second', model='vendor/remote')
    await asyncio.sleep(0)  # let the worker take the first call
    summary = await shadow.finish()
    assert summary[shadow.STATUS_PENDING] == 2
    records = _records(shadow_on)
    assert [r['status'] for r in records] == [shadow.STATUS_PENDING] * 2
    assert {r['prompt'].split('\n')[0] for r in records} == {'first', 'second'}


async def test_without_shadow_mode_no_confidence_is_asked_and_nothing_is_cut_or_shadowed(monkeypatch):
    '''2026-11-12: the confidence is asked only in shadow mode; it no longer changes production prompts.'''
    monkeypatch.setattr(config, 'SHADOW_MODEL', None)
    calls = _servers(monkeypatch, 'yes\nCONFIDENCE: 4', 'unused')
    reply = await triage.chat_openrouter('Q?', model='vendor/remote')
    assert calls[0][1]['messages'][-1]['content'] == 'Q?'
    assert reply.content == 'yes\nCONFIDENCE: 4' and reply.confidence is None
    assert len(calls) == 1, 'no local call'
    assert await shadow.finish() is None


SCHEMA = {'type': 'object', 'properties': {'rating': {'type': 'integer', 'minimum': 1, 'maximum': 5}},
          'required': ['rating']}


def _tool_call(arguments: dict | str, name: str = 'submit_x') -> list[dict]:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return [{'id': 'c1', 'type': 'function', 'function': {'name': name, 'arguments': raw}}]


async def test_a_forced_call_without_shadow_mode_offers_no_confidence_argument(monkeypatch):
    monkeypatch.setattr(config, 'SHADOW_MODEL', None)
    calls = _servers(monkeypatch, None, 'unused', remote_tool_calls=_tool_call({'rating': 4}))
    arguments, reply = await triage.chat_openrouter_tool('Rate this.', 'submit_x', 'Submit.', SCHEMA, model='vendor/remote')
    assert arguments == {'rating': 4} and reply.confidence is None
    sent = calls[0][1]
    assert sent['tool_choice'] == {'type': 'function', 'function': {'name': 'submit_x'}}
    assert sent['tools'][0]['function']['parameters'] == SCHEMA
    assert sent['messages'][-1]['content'] == 'Rate this.'
    assert len(calls) == 1


async def test_a_forced_call_in_shadow_mode_takes_its_confidence_as_an_argument_and_asks_the_local_model_in_text(
        monkeypatch, shadow_on):
    calls = _servers(monkeypatch, None, '{"rating": 3}\nCONFIDENCE: 2',
                     remote_tool_calls=_tool_call({'rating': 4, shadow.CONFIDENCE_KEY: 5}))
    arguments, reply = await triage.chat_openrouter_tool('Rate this.', 'submit_x', 'Submit.', SCHEMA, model='vendor/remote')
    assert arguments == {'rating': 4} and reply.confidence == 5
    offered = calls[0][1]['tools'][0]['function']['parameters']
    assert offered['properties'][shadow.CONFIDENCE_KEY] == shadow.CONFIDENCE_PROPERTY
    assert offered['required'] == ['rating'], 'the confidence is optional'
    await _settle()
    await shadow.finish()
    [record] = _records(shadow_on)
    assert record['remote']['content'] == '{"rating": 4}' and record['remote']['confidence'] == 5
    assert record['shadow']['content'] == '{"rating": 3}' and record['shadow']['confidence'] == 2
    local_prompt = calls[1][1]['prompt']
    assert local_prompt.startswith('Rate this.') and 'Instead of calling submit_x' in local_prompt
    assert shadow.CONFIDENCE_KEY not in local_prompt.split('JSON schema:')[1].split('\n')[0]


@pytest.mark.parametrize(('content', 'tool_calls', 'problem'), [
    ('{"rating": 4}', None, 'answered in text and called no function'),
    (None, _tool_call({'rating': 4}, name='other'), "0 calls to submit_x among ['other']"),
    (None, _tool_call('{"rating": 4'), 'arguments are not JSON'),
    (None, _tool_call({'rating': 9}), 'arguments break the schema at rating'),
    (None, _tool_call({}), "'rating' is a required property"),
])
async def test_an_unusable_forced_call_raises_with_what_is_wrong_who_served_it_and_the_arguments(
        monkeypatch, content, tool_calls, problem):
    monkeypatch.setattr(config, 'SHADOW_MODEL', None)
    _servers(monkeypatch, content, 'unused', remote_tool_calls=tool_calls)
    with pytest.raises(triage.UnusableToolCallError) as caught:
        await triage.chat_openrouter_tool('Rate this.', 'submit_x', 'Submit.', SCHEMA, model='vendor/remote')
    message = str(caught.value)
    assert problem in message and 'provider=P' in message and 'arguments:' in message
    assert caught.value.cost_usd == pytest.approx(0.001)
