'''Single-turn endpoint pin: drop the slowest and priciest thirds, pin the cheapest left.'''

import json

import pytest
from agentic_job_search import endpoint_pin, extract_openrouter, triage


def _endpoint(tag, bits, tps, prompt, completion, status=0):
    return {'tag': tag, 'quantization': f'fp{bits}' if bits else 'unknown', 'bits': bits,
            'throughput_p50': tps, 'prompt_price': prompt, 'completion_price': completion,
            'supports_tools': True, 'status': status}


# deepseek/deepseek-v4.1-flash-1017 as list_endpoints returned it on 2026-10-18 (trimmed).
LISTING_2026_10_18 = {'ok': True, 'endpoints': [
    _endpoint('relace', None, 138, 5e-8, 6e-7),
    _endpoint('morph/fp8', 8, 94, 5.9e-8, 5.5e-7),
    _endpoint('deepinfra/fp8/flex', 8, 21, 7.84e-8, 2.352e-7),
    _endpoint('inference-net/fp8', 8, 184, 8e-8, 5e-7),
    _endpoint('deepinfra/fp8', 8, 87, 9.8e-8, 2.94e-7),
    _endpoint('deepinfra/fp8/priority', 8, 86, 1.47e-7, 4.41e-7),
    _endpoint('wafer/mxfp4', 4, 112, 2e-7, 8e-7),
    _endpoint('crusoe/fp8', 8, 119, 2.9e-7, 1.2e-6),
    _endpoint('parasail/fp8', 8, 124, 3e-7, 1.2e-6),
    _endpoint('deepseek', None, 54, 3e-7, 1.2e-6),
]}


@pytest.fixture(autouse=True)
def _no_pins():
    endpoint_pin.reset_pins()
    yield
    endpoint_pin.reset_pins()


def test_selection_drops_the_slowest_and_priciest_thirds_and_ranks_by_price():
    kept, report = endpoint_pin.select_single_turn_endpoints(LISTING_2026_10_18)
    # 7 usable fp8 endpoints (unknown bits and the 4-bit one excluded): 2 slowest and 2 priciest go.
    assert [e['tag'] for _p, e in kept] == ['deepinfra/fp8', 'morph/fp8', 'inference-net/fp8']
    assert '7 usable of 10' in report
    assert "'deepinfra/fp8/flex'" in report and "'parasail/fp8'" in report


def test_selection_excludes_down_and_unmeasured_endpoints():
    listing = {'endpoints': [_endpoint('down/fp8', 8, 100, 1e-7, 1e-7, status=-2),
                             _endpoint('unmeasured/fp8', 8, None, 1e-7, 1e-7),
                             _endpoint('ok/fp8', 8, 50, 1e-7, 1e-7)]}
    kept, _report = endpoint_pin.select_single_turn_endpoints(listing)
    assert [e['tag'] for _p, e in kept] == ['ok/fp8'], 'fewer than three usable drops nothing'


async def _pin_from(monkeypatch, listing):
    async def fake_call_mcp_tool(url, tool_name, args, timeout_seconds=None):
        assert tool_name == 'list_endpoints'
        return json.dumps(listing)

    monkeypatch.setattr(endpoint_pin, 'call_mcp_tool', fake_call_mcp_tool)
    await endpoint_pin.pin_single_turn_endpoints('m/1')


async def test_a_failure_on_the_pin_moves_to_the_next_cheapest(monkeypatch, caplog):
    await _pin_from(monkeypatch, LISTING_2026_10_18)
    routing = endpoint_pin.routing_for('m/1')
    assert routing == {'provider': 'deepinfra/fp8', 'quantization': 'fp8'}
    endpoint_pin.endpoint_failed('m/1', routing, 'timed out after 180.0s')
    assert endpoint_pin.routing_for('m/1')['provider'] == 'morph/fp8'
    endpoint_pin.endpoint_failed('m/1', routing, 'a late second report of the same failure')
    assert endpoint_pin.routing_for('m/1')['provider'] == 'morph/fp8', 'a stale report must not drop the new pin'
    assert any('switched to morph/fp8' in r.getMessage() for r in caplog.records)


async def test_a_listing_failure_leaves_the_model_unpinned(monkeypatch):
    await _pin_from(monkeypatch, {'ok': False, 'error': 'boom'})
    assert endpoint_pin.routing_for('m/1') == {}


async def test_chat_openrouter_serves_on_the_pin_and_reports_a_timeout(monkeypatch):
    await _pin_from(monkeypatch, LISTING_2026_10_18)
    sent = []

    async def timing_out(url, tool_name, args, timeout_seconds=None):
        sent.append(args)
        raise RuntimeError("MCP tool 'chat' timed out after 180.0s")

    monkeypatch.setattr(triage, 'call_mcp_tool', timing_out)
    with pytest.raises(RuntimeError):
        await triage.chat_openrouter('hi', model='m/1')
    assert sent[0]['provider'] == 'deepinfra/fp8' and sent[0]['quantization'] == 'fp8'
    assert endpoint_pin.routing_for('m/1')['provider'] == 'morph/fp8'


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
    await _pin_from(monkeypatch, LISTING_2026_10_18)
    monkeypatch.setattr(extract_openrouter, 'MODEL_NAME_EXTRACTOR', 'm/1')
    sent = []

    async def capture(url, tool_name, args, timeout_seconds=None):
        sent.append(args)
        raise RuntimeError('timed out')

    monkeypatch.setattr(extract_openrouter, 'call_mcp_tool', capture)
    candidate = {'job_id': '1', 'company': 'Acme', 'title': 'Eng', 'url': 'u', 'date_posted': '', 'snippet': ''}
    assert await extract_openrouter.extract_job_page_openrouter(
        candidate, 'text', {'cost': 0.0, 'input_tokens': 0, 'output_tokens': 0}, 'sys') is None
    assert sent[0]['provider'] == 'deepinfra/fp8'
    assert endpoint_pin.routing_for('m/1')['provider'] == 'morph/fp8'
