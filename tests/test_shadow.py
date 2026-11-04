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


def _servers(monkeypatch, remote_content: str, local: str | Exception, local_gate: asyncio.Event | None = None):
    '''Fake OpenRouter and local servers behind triage.call_mcp_tool; returns the calls made.'''
    calls: list[tuple[str, dict]] = []

    async def fake_call(url, tool, args, timeout_seconds=None):
        calls.append((tool, args))
        if url == config.LLM_OPENROUTER_MCP_URL:
            return json.dumps({'ok': True, 'content': remote_content, 'cost_usd': 0.001, 'model': 'vendor/remote',
                               'provider': 'P', 'finish_reason': 'stop', 'usage': {}})
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


async def test_shadow_mode_off_changes_nothing(monkeypatch):
    monkeypatch.setattr(config, 'SHADOW_MODEL', None)
    calls = _servers(monkeypatch, 'yes\nCONFIDENCE: 4', 'unused')
    reply = await triage.chat_openrouter('Q?', model='vendor/remote')
    assert calls[0][1]['messages'][-1]['content'] == 'Q?'
    assert reply.content == 'yes\nCONFIDENCE: 4' and reply.confidence is None
    assert len(calls) == 1
    assert await shadow.finish() is None
