'''Single-turn endpoint pin: the tool server ranks the endpoints, the pin walks the list it returns.

The ranking rule itself (drop the slowest and priciest shares, take the cheapest survivor) lives in
tools_llm_remote_openrouter's `rank_providers`, so what is tested here is only the caller's half:
the mix and floor it asks with, and the per-run pin/failure state machine.
'''

import json

import httpx
import pytest
from agentic_job_search import endpoint_pin, extract_openrouter, triage
from agentic_job_search.config import (
    SINGLE_TURN_CACHED_PROMPT_SHARE,
    SINGLE_TURN_COMPLETION_TOKENS_PER_PROMPT_TOKEN,
    SINGLE_TURN_DROP_PRICIEST_SHARE,
    SINGLE_TURN_DROP_SLOWEST_SHARE,
    SINGLE_TURN_MIN_BITS_PER_WEIGHT,
)

# The ranked answer the tool server returned on 2026-10-18 for deepseek/deepseek-v4.1-flash-1017:
# the three survivors of the slowest/priciest cuts, cheapest first.
RANKED_2026_10_18 = {'ok': True, 'model': 'm/1', 'report': '7 usable of 10; kept deepinfra/fp8',
                     'ranked': [
                         {'tag': 'deepinfra/fp8', 'quantization': 'fp8',
                          'effective_price_per_million': 2.156e-7},
                         {'tag': 'morph/fp8', 'quantization': 'fp8',
                          'effective_price_per_million': 2.79e-7},
                         {'tag': 'inference-net/fp8', 'quantization': 'fp8',
                          'effective_price_per_million': 2.8e-7},
                     ]}


@pytest.fixture(autouse=True)
def _no_pins():
    endpoint_pin.reset_pins()
    yield
    endpoint_pin.reset_pins()


async def _pin_from(monkeypatch, response, sent=None):
    async def fake_call_mcp_tool(url, tool_name, args, timeout_seconds=None):
        assert tool_name == 'rank_providers', "the ranking rule is the tool server's to apply"
        if sent is not None:
            sent.append(args)
        return json.dumps(response)

    monkeypatch.setattr(endpoint_pin, 'call_mcp_tool', fake_call_mcp_tool)
    await endpoint_pin.pin_single_turn_endpoints('m/1')


async def test_the_pin_takes_the_first_ranked_endpoint(monkeypatch):
    await _pin_from(monkeypatch, RANKED_2026_10_18)
    assert endpoint_pin.routing_for('m/1') == {'provider': 'deepinfra/fp8', 'quantization': 'fp8'}


async def test_the_pin_asks_for_the_single_turn_mix_and_floor(monkeypatch):
    '''The caller supplies only its own numbers — the mix and the precision floor.'''
    sent = []
    await _pin_from(monkeypatch, RANKED_2026_10_18, sent)
    assert sent == [{
        'model': 'm/1',
        'min_bits': SINGLE_TURN_MIN_BITS_PER_WEIGHT,
        'cached_prompt_share': SINGLE_TURN_CACHED_PROMPT_SHARE,
        'completion_tokens_per_prompt_token': SINGLE_TURN_COMPLETION_TOKENS_PER_PROMPT_TOKEN,
        'drop_slowest_share': SINGLE_TURN_DROP_SLOWEST_SHARE,
        'drop_priciest_share': SINGLE_TURN_DROP_PRICIEST_SHARE,
    }]


async def test_a_failure_on_the_pin_moves_to_the_next_cheapest(monkeypatch, caplog):
    await _pin_from(monkeypatch, RANKED_2026_10_18)
    routing = endpoint_pin.routing_for('m/1')
    assert endpoint_pin.endpoint_failed('m/1', routing, 'timed out after 180.0s'), 'the pin moved: ask again'
    assert endpoint_pin.routing_for('m/1')['provider'] == 'morph/fp8'
    assert endpoint_pin.endpoint_failed('m/1', routing, 'a late second report of the same failure')
    assert endpoint_pin.routing_for('m/1')['provider'] == 'morph/fp8', 'a stale report must not drop the new pin'
    assert any('switched to morph/fp8' in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize('error', [
    httpx.ConnectError('MCP initialize at http://127.0.0.1:8006/mcp transport failed: All connection attempts failed'),
    httpx.RemoteProtocolError('peer closed connection without sending complete message body'),
])
async def test_a_failure_to_reach_the_tool_server_keeps_the_pin(monkeypatch, error):
    '''2026-10-20: the tool server died and each refused connection dropped one more endpoint.'''
    await _pin_from(monkeypatch, RANKED_2026_10_18)
    for _ in range(3):
        assert not endpoint_pin.endpoint_failed('m/1', endpoint_pin.routing_for('m/1'), error)
    assert endpoint_pin.routing_for('m/1')['provider'] == 'deepinfra/fp8'


async def test_a_read_timeout_is_the_endpoints_and_drops_it(monkeypatch):
    await _pin_from(monkeypatch, RANKED_2026_10_18)
    endpoint_pin.endpoint_failed('m/1', endpoint_pin.routing_for('m/1'), httpx.ReadTimeout('timed out'))
    assert endpoint_pin.routing_for('m/1')['provider'] == 'morph/fp8'


async def test_chat_openrouter_keeps_the_pin_when_the_tool_server_is_down(monkeypatch):
    await _pin_from(monkeypatch, RANKED_2026_10_18)

    async def refused(url, tool_name, args, timeout_seconds=None):
        raise httpx.ConnectError('All connection attempts failed')

    monkeypatch.setattr(triage, 'call_mcp_tool', refused)
    with pytest.raises(httpx.ConnectError):
        await triage.chat_openrouter('hi', model='m/1')
    assert endpoint_pin.routing_for('m/1')['provider'] == 'deepinfra/fp8'


async def test_an_ok_false_reply_drops_the_endpoint_and_asks_the_next_once(monkeypatch):
    await _pin_from(monkeypatch, RANKED_2026_10_18)
    sent = []

    async def refusing(url, tool_name, args, timeout_seconds=None):
        sent.append(args.get('provider'))
        return json.dumps({'ok': False, 'error': 'provider returned 503'})

    monkeypatch.setattr(triage, 'call_mcp_tool', refusing)
    with pytest.raises(RuntimeError, match='provider returned 503'):
        await triage.chat_openrouter('hi', model='m/1')
    assert sent == ['deepinfra/fp8', 'morph/fp8'], 'asked once more on the new pin, then the caller falls back'
    assert endpoint_pin.routing_for('m/1')['provider'] == 'inference-net/fp8'


CHAT_OK = {'ok': True, 'content': 'fine', 'cost_usd': 0.0001, 'model': 'm/1', 'provider': 'Morph',
           'finish_reason': 'stop', 'usage': {}}


async def test_a_429_on_the_pin_is_answered_by_the_next_provider(monkeypatch):
    '''2026-10-26: a 429 on inference-net/us moved the pin to morph, but that call went to the
    Anthropic fallback instead of being asked of morph.'''
    ranked = {**RANKED_2026_10_18, 'ranked': [
        {'tag': 'inference-net/us', 'quantization': 'fp8', 'effective_price_per_million': 2.8e-7},
        {'tag': 'inference-net/fp8', 'quantization': 'fp8', 'effective_price_per_million': 2.8e-7},
        {'tag': 'morph/fp8', 'quantization': 'fp8', 'effective_price_per_million': 2.9e-7},
    ]}
    await _pin_from(monkeypatch, ranked)
    sent = []

    async def throttled_then_ok(url, tool_name, args, timeout_seconds=None):
        sent.append(args.get('provider'))
        if args.get('provider') == 'inference-net/us':
            return json.dumps({'ok': False, 'error': INFERENCE_NET_429})
        return json.dumps(CHAT_OK)

    monkeypatch.setattr(triage, 'call_mcp_tool', throttled_then_ok)
    reply = await triage.chat_openrouter('hi', model='m/1')
    assert reply.content == 'fine'
    assert sent == ['inference-net/us', 'morph/fp8']


async def test_a_tool_server_failure_is_not_asked_again(monkeypatch):
    await _pin_from(monkeypatch, RANKED_2026_10_18)
    sent = []

    async def refused(url, tool_name, args, timeout_seconds=None):
        sent.append(args.get('provider'))
        raise httpx.ConnectError('All connection attempts failed')

    monkeypatch.setattr(triage, 'call_mcp_tool', refused)
    with pytest.raises(httpx.ConnectError):
        await triage.chat_openrouter('hi', model='m/1')
    assert sent == ['deepinfra/fp8']


async def test_an_unpinned_failure_is_not_asked_again(monkeypatch):
    sent = []

    async def refusing(url, tool_name, args, timeout_seconds=None):
        sent.append(args.get('provider'))
        return json.dumps({'ok': False, 'error': 'provider returned 503'})

    monkeypatch.setattr(triage, 'call_mcp_tool', refusing)
    with pytest.raises(RuntimeError):
        await triage.chat_openrouter('hi', model='other/model')
    assert sent == [None]


async def test_a_ranking_failure_leaves_the_model_unpinned(monkeypatch):
    await _pin_from(monkeypatch, {'ok': False, 'error': 'boom'})
    assert endpoint_pin.routing_for('m/1') == {}


async def test_an_empty_ranking_leaves_the_model_unpinned(monkeypatch):
    await _pin_from(monkeypatch, {'ok': True, 'model': 'm/1', 'report': 'none', 'ranked': []})
    assert endpoint_pin.routing_for('m/1') == {}


async def test_chat_openrouter_serves_on_the_pin_and_reports_a_timeout(monkeypatch):
    await _pin_from(monkeypatch, RANKED_2026_10_18)
    sent = []

    async def timing_out(url, tool_name, args, timeout_seconds=None):
        sent.append(args)
        raise RuntimeError("MCP tool 'chat' timed out after 180.0s")

    monkeypatch.setattr(triage, 'call_mcp_tool', timing_out)
    with pytest.raises(RuntimeError):
        await triage.chat_openrouter('hi', model='m/1')
    assert sent[0]['provider'] == 'deepinfra/fp8' and sent[0]['quantization'] == 'fp8'
    assert [a['provider'] for a in sent] == ['deepinfra/fp8', 'morph/fp8']
    assert endpoint_pin.routing_for('m/1')['provider'] == 'inference-net/fp8'


async def test_an_unpinned_model_sends_no_provider(monkeypatch):
    sent = []

    async def capture(url, tool_name, args, timeout_seconds=None):
        sent.append(args)
        raise RuntimeError('stop here')

    monkeypatch.setattr(triage, 'call_mcp_tool', capture)
    with pytest.raises(RuntimeError):
        await triage.chat_openrouter('hi', model='other/model')
    assert 'provider' not in sent[0]


async def test_extract_loop_serves_on_the_pin(monkeypatch):
    await _pin_from(monkeypatch, RANKED_2026_10_18)
    monkeypatch.setattr(extract_openrouter, 'MODEL_NAME_EXTRACTOR', 'm/1')
    sent = []

    async def capture(url, tool_name, args, timeout_seconds=None):
        sent.append(args)
        raise RuntimeError('timed out')

    monkeypatch.setattr(extract_openrouter, 'call_mcp_tool', capture)
    candidate = {'job_id': '1', 'company': 'Acme', 'title': 'Eng', 'url': 'u', 'date_posted': '', 'snippet': ''}
    assert await extract_openrouter.extract_job_page_openrouter(
        candidate, 'text', {'cost': 0.0, 'input_tokens': 0, 'output_tokens': 0}, 'sys') is None
    assert [a['provider'] for a in sent] == ['deepinfra/fp8', 'morph/fp8'], 'asked once more on the new pin'
    assert endpoint_pin.routing_for('m/1')['provider'] == 'inference-net/fp8'


INFERENCE_NET_429 = ('chat failed: HTTPError: 429 Client Error: Too Many Requests for url: '
                     'https://openrouter.ai/api/v1/chat/completions — body: {"error":{"message":'
                     '"Provider returned error","code":429,"metadata":{"provider_name":"InferenceNet",'
                     '"limit_source":"upstream_provider_shared_pool"}}}')


async def test_a_429_drops_every_endpoint_of_that_provider(monkeypatch, caplog):
    '''2026-10-22: InferenceNet's shared pool throttled inference-net/us, and the pin's next pick,
    inference-net/fp8, was the same pool and failed the same way.'''
    ranked = {**RANKED_2026_10_18, 'ranked': [
        {'tag': 'inference-net/us', 'quantization': 'fp8', 'effective_price_per_million': 2.8e-7},
        {'tag': 'inference-net/fp8', 'quantization': 'fp8', 'effective_price_per_million': 2.8e-7},
        {'tag': 'gmicloud/fp8', 'quantization': 'fp8', 'effective_price_per_million': 4.68e-7},
    ]}
    await _pin_from(monkeypatch, ranked)
    endpoint_pin.endpoint_failed('m/1', endpoint_pin.routing_for('m/1'), INFERENCE_NET_429)
    assert endpoint_pin.routing_for('m/1')['provider'] == 'gmicloud/fp8'
    assert any('inference-net is rate-limited' in r.getMessage() and 'inference-net/fp8' in r.getMessage()
               for r in caplog.records)


async def test_a_non_429_failure_drops_only_that_endpoint(monkeypatch):
    ranked = {**RANKED_2026_10_18, 'ranked': [
        {'tag': 'inference-net/us', 'quantization': 'fp8', 'effective_price_per_million': 2.8e-7},
        {'tag': 'inference-net/fp8', 'quantization': 'fp8', 'effective_price_per_million': 2.8e-7},
    ]}
    await _pin_from(monkeypatch, ranked)
    endpoint_pin.endpoint_failed('m/1', endpoint_pin.routing_for('m/1'), 'timed out after 180.0s')
    assert endpoint_pin.routing_for('m/1')['provider'] == 'inference-net/fp8'
