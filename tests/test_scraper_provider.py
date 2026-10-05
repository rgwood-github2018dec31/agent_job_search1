"""The scraper's per-run OpenRouter endpoint pin (scrape_openrouter.ProviderPin).

On 2026-09-23 the six scraper queries cost $0.010-0.022 each; on 2026-09-24 the same queries, with
the same token counts and cache hit rates, cost $0.19-0.37 — OpenRouter had drawn fp8 providers
charging 30-40x more for cache reads. The prices below are the live ones from that day.
"""
import json
import logging

import pytest
from agentic_job_search import agent
from agentic_job_search import scrape_openrouter as so
from agentic_job_search import tools_generic as tools
from agentic_job_search.config import (
    SCRAPER_COST_ALERT_FACTOR,
    SCRAPER_MAX_COMPLETION_TOKENS_PER_ITERATION,
    SCRAPER_MIN_BITS_PER_WEIGHT,
    SCRAPER_PROVIDER_MAX_SWITCHES,
    SCRAPER_VERBOSE_MIN_ITERATIONS,
)
from agentic_job_search.triage import ProviderUnavailableError

# Imported at collection time, before conftest's autouse stub replaces it on the module.
REAL_PIN_SCRAPER_PROVIDER = so.pin_scraper_provider

MODEL = 'deepseek/deepseek-v4-flash-0731'
MIN_BITS = SCRAPER_MIN_BITS_PER_WEIGHT[MODEL]


def _endpoint(tag, prompt, cache, completion, quantization='fp8', bits=8, status=0, tools_ok=True):
    return {'tag': tag, 'provider_name': tag.split('/')[0], 'quantization': quantization, 'bits': bits,
            'prompt_price': prompt, 'cache_read_price': cache, 'completion_price': completion,
            'status': status, 'supports_tools': tools_ok}


def _listing(*endpoints):
    return {'ok': True, 'model': MODEL, 'default_quantization': 'fp8', 'closed_weights': False,
            'endpoints': list(endpoints)}


# Live 2026-09-24 prices, per token.
STREAMLAKE = _endpoint('streamlake/fp8', 5.28e-8, 1.68e-9, 1.584e-7)
DEEPINFRA = _endpoint('deepinfra/fp8', 6e-8, 1.5e-8, 1.8e-7)
COREWEAVE = _endpoint('coreweave/fp8', 1.3e-7, 7e-8, 2.8e-7)
NO_CACHE_PRICE = _endpoint('mancer/fp8', 5e-8, None, 1.6e-7)


# --- ranking ---------------------------------------------------------------------------------------

def test_ranking_is_by_the_scrapers_token_mix_not_the_prompt_price():
    """NO_CACHE_PRICE has the lowest prompt price, but at 94% cache reads it is the dearest."""
    ranked = [e['tag'] for _, e in so.rank_endpoints(_listing(COREWEAVE, NO_CACHE_PRICE, DEEPINFRA, STREAMLAKE), [], MIN_BITS)]
    assert ranked == ['streamlake/fp8', 'deepinfra/fp8', 'mancer/fp8', 'coreweave/fp8']


def test_the_estimate_reproduces_the_expensive_run():
    """CoreWeave's estimate lands on the $0.2566 the 'AI Architect' query cost on 2026-09-24."""
    prompt_tokens = 3_471_892
    estimate = so.effective_price_per_million(COREWEAVE) * prompt_tokens / so.PER_MILLION
    assert estimate == pytest.approx(0.2566, rel=0.1)


def test_a_missing_price_is_never_guessed():
    assert so.effective_price_per_million({**STREAMLAKE, 'completion_price': None}) is None
    assert so.effective_price_per_million({**STREAMLAKE, 'prompt_price': None}) is None


@pytest.mark.parametrize('unusable', [
    _endpoint('down/fp8', 1e-9, 1e-10, 1e-9, status=-2),
    _endpoint('notools/fp8', 1e-9, 1e-10, 1e-9, tools_ok=None),
    _endpoint('lowbits/fp4', 1e-9, 1e-10, 1e-9, quantization='fp4', bits=4),
    {**_endpoint('x/fp8', 1e-9, 1e-10, 1e-9), 'tag': None},
    _endpoint('unknown', 1e-9, 1e-10, 1e-9, quantization='unknown', bits=None),
])
def test_unusable_endpoints_are_never_chosen_however_cheap(unusable):
    ranked = so.rank_endpoints(_listing(unusable, DEEPINFRA), [], MIN_BITS)
    assert [e['tag'] for _, e in ranked] == ['deepinfra/fp8']


def test_a_precision_above_the_servers_default_is_still_a_candidate():
    """2026-09-27: the default became bf16 and every fp8 endpoint vanished from the list."""
    bf16 = _endpoint('morph/bf16', 1.4e-7, 3.6e-8, 4e-7, quantization='bf16', bits=16)
    listing = {**_listing(bf16, DEEPINFRA, STREAMLAKE), 'default_quantization': 'bf16'}
    assert [e['tag'] for _, e in so.rank_endpoints(listing, [], MIN_BITS)] == \
        ['streamlake/fp8', 'deepinfra/fp8', 'morph/bf16']


def test_a_tag_listed_twice_is_one_candidate():
    assert [e['tag'] for _, e in so.rank_endpoints(_listing(DEEPINFRA, DEEPINFRA), [], MIN_BITS)] == ['deepinfra/fp8']


def test_a_model_without_a_precision_floor_fails_fast():
    with pytest.raises(KeyError, match='SCRAPER_MIN_BITS_PER_WEIGHT'):
        so.min_bits_for('someone/unlisted-model')


async def test_pinning_a_model_without_a_precision_floor_raises_rather_than_running_unpinned(monkeypatch):
    _wire(monkeypatch, _listing(STREAMLAKE))
    with pytest.raises(KeyError, match='unlisted-model'):
        await REAL_PIN_SCRAPER_PROVIDER('someone/unlisted-model')


def test_failed_endpoints_are_excluded():
    assert [e['tag'] for _, e in so.rank_endpoints(_listing(STREAMLAKE, DEEPINFRA), ['streamlake/fp8'], MIN_BITS)] == ['deepinfra/fp8']


# --- choosing --------------------------------------------------------------------------------------

def _wire(monkeypatch, listing, chat_responses=None, sent=None):
    """Route call_mcp_tool: list_endpoints -> the listing; chat -> the next canned response."""
    chat_responses = list(chat_responses or [])

    async def fake_call(url, tool, args, timeout_seconds=None):
        if sent is not None:
            sent.append((tool, json.loads(json.dumps(args))))
        if tool == 'list_endpoints':
            return json.dumps(listing)
        return chat_responses.pop(0)

    monkeypatch.setattr(so, 'call_mcp_tool', fake_call)


async def test_the_cheapest_usable_endpoint_is_pinned(monkeypatch):
    _wire(monkeypatch, _listing(COREWEAVE, DEEPINFRA, STREAMLAKE))
    pin = await REAL_PIN_SCRAPER_PROVIDER(MODEL)
    assert pin.tag == 'streamlake/fp8'
    assert pin.estimate_per_million == pytest.approx(so.effective_price_per_million(STREAMLAKE))


async def test_a_failed_listing_leaves_the_run_unpinned_and_says_so(monkeypatch, caplog):
    _wire(monkeypatch, {'ok': False, 'error': 'HTTP 503'})
    monkeypatch.setattr(tools, '_ui_alerts', [])
    with caplog.at_level(logging.WARNING):
        pin = await REAL_PIN_SCRAPER_PROVIDER(MODEL)
    assert pin.tag is None
    assert [a['kind'] for a in tools._ui_alerts] == ['provider_unpinned']
    assert any('running unpinned' in r.message and 'HTTP 503' in r.message for r in caplog.records)


# --- the scraper loop ------------------------------------------------------------------------------

def _chat(ok=True, error=None, provider='StreamLake', cost=0.001, prompt=1000, cached=900):
    data = {'ok': ok, 'content': 'done', 'tool_calls': None, 'finish_reason': 'stop', 'provider': provider,
            'usage': {'prompt_tokens': prompt, 'completion_tokens': 10,
                      'prompt_tokens_details': {'cached_tokens': cached}}, 'cost_usd': cost}
    if error:
        data = {'ok': False, 'error': error}
    return json.dumps(data)


async def _no_browser(name, args):
    raise AssertionError('no browser call expected')


async def _pinned(monkeypatch, listing):
    _wire(monkeypatch, listing)
    return await REAL_PIN_SCRAPER_PROVIDER(MODEL)


async def test_every_call_of_the_run_goes_to_the_pinned_endpoint(monkeypatch):
    pin = await _pinned(monkeypatch, _listing(DEEPINFRA, STREAMLAKE))
    sent = []
    _wire(monkeypatch, _listing(DEEPINFRA, STREAMLAKE), [_chat(), _chat()], sent)
    for query in ('AI Architect', 'Staff AI Engineer'):
        session = so.ScrapeSession(_no_browser, query, provider_pin=pin)
        await session.run('sys', 'user', [])
        assert session.served_by == {'StreamLake'}
    assert [args['provider'] for tool, args in sent if tool == 'chat'] == ['streamlake/fp8', 'streamlake/fp8']


async def test_a_provider_failure_switches_to_the_next_cheapest_for_the_rest_of_the_run(monkeypatch, caplog):
    pin = await _pinned(monkeypatch, _listing(DEEPINFRA, STREAMLAKE))
    monkeypatch.setattr(tools, '_ui_alerts', [])
    monkeypatch.setattr(so, 'TRANSIENT_RETRY_DELAY_SECONDS', 0)
    sent = []
    down = _chat(error='OpenRouter chat failed: 503 Server Error: Service Unavailable for url: https://openrouter.ai/api/v1/chat/completions')
    _wire(monkeypatch, _listing(DEEPINFRA, STREAMLAKE), [down, down, _chat(provider='DeepInfra'), _chat(provider='DeepInfra')], sent)

    with caplog.at_level(logging.WARNING):
        await so.ScrapeSession(_no_browser, 'AI Architect', provider_pin=pin).run('sys', 'user', [])
        await so.ScrapeSession(_no_browser, 'Staff AI Engineer', provider_pin=pin).run('sys', 'user', [])

    chats = [args['provider'] for tool, args in sent if tool == 'chat']
    assert chats == ['streamlake/fp8', 'streamlake/fp8', 'deepinfra/fp8', 'deepinfra/fp8'], \
        'one transient retry, then the switch, and the next query stays on the new endpoint'
    assert pin.failed == ['streamlake/fp8'] and pin.switches == 1
    assert [a['kind'] for a in tools._ui_alerts] == ['provider_switch']
    assert any('switched to deepinfra/fp8' in r.message for r in caplog.records)


async def test_a_429_on_a_pinned_endpoint_is_that_providers_rate_limit(monkeypatch):
    pin = await _pinned(monkeypatch, _listing(DEEPINFRA, STREAMLAKE))
    monkeypatch.setattr(tools, '_ui_alerts', [])
    sent = []
    bf16 = _endpoint('morph/bf16', 1.4e-7, 3.6e-8, 4e-7, quantization='bf16', bits=16)
    _wire(monkeypatch, _listing(DEEPINFRA, STREAMLAKE, bf16), [_chat(error='429 Client Error: Too Many Requests for url: x'), _chat()], sent)
    await so.ScrapeSession(_no_browser, 'q', provider_pin=pin).run('sys', 'user', [])
    assert pin.tag == 'deepinfra/fp8'
    chats = [(args['provider'], args['quantization']) for tool, args in sent if tool == 'chat']
    assert chats == [('streamlake/fp8', 'fp8'), ('deepinfra/fp8', 'fp8')], 'each call names its endpoint AND its precision'


@pytest.mark.parametrize('status', [401, 402, 403])
async def test_an_account_failure_never_switches_and_still_aborts(monkeypatch, status):
    pin = await _pinned(monkeypatch, _listing(DEEPINFRA, STREAMLAKE))
    _wire(monkeypatch, _listing(DEEPINFRA, STREAMLAKE), [_chat(error=f'{status} Client Error: account problem for url: x')])
    with pytest.raises(ProviderUnavailableError):
        await so.ScrapeSession(_no_browser, 'q', provider_pin=pin).run('sys', 'user', [])
    assert pin.tag == 'streamlake/fp8' and pin.switches == 0


async def test_switching_stops_at_the_cap(monkeypatch):
    endpoints = [_endpoint(f'p{i}/fp8', 1e-8 * (i + 1), 1e-9, 1e-7) for i in range(SCRAPER_PROVIDER_MAX_SWITCHES + 2)]
    pin = await _pinned(monkeypatch, _listing(*endpoints))
    monkeypatch.setattr(tools, '_ui_alerts', [])
    # The server's refusal for an endpoint that vanished carries no HTTP status: still a switch.
    gone = _chat(error='provider pin failed: provider has no endpoint for the model')
    # Each query up to the cap fails twice (pinned endpoint, then its replacement); the last fails once.
    _wire(monkeypatch, _listing(*endpoints), [gone] * (2 * SCRAPER_PROVIDER_MAX_SWITCHES + 1))
    for _ in range(SCRAPER_PROVIDER_MAX_SWITCHES + 1):
        with pytest.raises(RuntimeError):
            await so.ScrapeSession(_no_browser, 'q', provider_pin=pin).run('sys', 'user', [])
    assert pin.switches == SCRAPER_PROVIDER_MAX_SWITCHES
    assert pin.failed == [f'p{i}/fp8' for i in range(SCRAPER_PROVIDER_MAX_SWITCHES + 1)]
    assert len(tools._ui_alerts) == SCRAPER_PROVIDER_MAX_SWITCHES


def _turn(completion_tokens: int, done: bool = False) -> str:
    '''One chat reply: a harmless local tool call, or the closing text when `done`.'''
    call = {'id': 'c', 'function': {'name': 'record_listings', 'arguments': '{}'}}
    return json.dumps({
        'ok': True, 'content': 'done' if done else None, 'tool_calls': None if done else [call],
        'finish_reason': 'stop', 'usage': {'prompt_tokens': 1000, 'completion_tokens': completion_tokens}})


async def test_an_endpoint_that_rambles_is_dropped_for_the_rest_of_the_run(monkeypatch):
    '''2026-10-02: open-inference/fp8 answered every call, at ~13x the usual output per turn, and
    never finished a search. Nothing counted that as a failure, so the run stayed on it for hours.'''
    pin = await _pinned(monkeypatch, _listing(DEEPINFRA, STREAMLAKE))
    monkeypatch.setattr(tools, '_ui_alerts', [])
    sent = []
    rambling = SCRAPER_MAX_COMPLETION_TOKENS_PER_ITERATION + 1
    _wire(monkeypatch, _listing(DEEPINFRA, STREAMLAKE),
          [_turn(rambling)] * SCRAPER_VERBOSE_MIN_ITERATIONS + [_turn(10, done=True)], sent)

    await so.ScrapeSession(_no_browser, 'q', provider_pin=pin).run('sys', 'user', [])

    providers = [args['provider'] for tool, args in sent if tool == 'chat']
    assert providers == ['streamlake/fp8'] * SCRAPER_VERBOSE_MIN_ITERATIONS + ['deepinfra/fp8']
    assert pin.failed == ['streamlake/fp8']
    assert [a['kind'] for a in tools._ui_alerts] == ['provider_switch']
    assert 'completion tokens per turn' in tools._ui_alerts[0]['detail']


async def test_an_endpoint_at_normal_output_is_kept(monkeypatch):
    pin = await _pinned(monkeypatch, _listing(DEEPINFRA, STREAMLAKE))
    at_the_limit = SCRAPER_MAX_COMPLETION_TOKENS_PER_ITERATION
    _wire(monkeypatch, _listing(DEEPINFRA, STREAMLAKE),
          [_turn(at_the_limit)] * (SCRAPER_VERBOSE_MIN_ITERATIONS + 2) + [_turn(10, done=True)])

    await so.ScrapeSession(_no_browser, 'q', provider_pin=pin).run('sys', 'user', [])

    assert pin.tag == 'streamlake/fp8' and pin.switches == 0


async def test_one_long_turn_early_in_a_pass_is_not_rambling(monkeypatch):
    '''The average is not judged before SCRAPER_VERBOSE_MIN_ITERATIONS turns.'''
    pin = await _pinned(monkeypatch, _listing(DEEPINFRA, STREAMLAKE))
    long_turn = SCRAPER_MAX_COMPLETION_TOKENS_PER_ITERATION * SCRAPER_VERBOSE_MIN_ITERATIONS
    _wire(monkeypatch, _listing(DEEPINFRA, STREAMLAKE), [_turn(long_turn), _turn(10, done=True)])

    await so.ScrapeSession(_no_browser, 'q', provider_pin=pin).run('sys', 'user', [])

    assert pin.switches == 0


async def test_a_chat_timeout_drops_the_pinned_endpoint_and_asks_the_next(monkeypatch):
    '''2026-10-02: three 300 s timeouts on one endpoint, each failing a query and changing nothing.'''
    pin = await _pinned(monkeypatch, _listing(DEEPINFRA, STREAMLAKE))
    monkeypatch.setattr(tools, '_ui_alerts', [])
    providers = []

    async def fake_call(url, tool, args, timeout_seconds=None):
        if tool == 'list_endpoints':
            return json.dumps(_listing(DEEPINFRA, STREAMLAKE))
        providers.append(args['provider'])
        if args['provider'] == 'streamlake/fp8':
            raise RuntimeError(f"MCP tool 'chat' at {url} timed out after {timeout_seconds}s")
        return _chat()

    monkeypatch.setattr(so, 'call_mcp_tool', fake_call)
    await so.ScrapeSession(_no_browser, 'q', provider_pin=pin).run('sys', 'user', [])

    assert providers == ['streamlake/fp8', 'deepinfra/fp8']
    assert pin.tag == 'deepinfra/fp8'
    assert [a['kind'] for a in tools._ui_alerts] == ['provider_switch']


async def test_a_chat_failure_that_is_not_a_timeout_still_raises(monkeypatch):
    pin = await _pinned(monkeypatch, _listing(DEEPINFRA, STREAMLAKE))

    async def fake_call(url, tool, args, timeout_seconds=None):
        raise RuntimeError('All connection attempts failed')

    monkeypatch.setattr(so, 'call_mcp_tool', fake_call)
    with pytest.raises(RuntimeError, match='connection attempts'):
        await so.ScrapeSession(_no_browser, 'q', provider_pin=pin).run('sys', 'user', [])
    assert pin.switches == 0


async def test_a_timeout_with_no_endpoint_left_still_fails_the_query(monkeypatch):
    pin = await _pinned(monkeypatch, _listing(STREAMLAKE))

    async def fake_call(url, tool, args, timeout_seconds=None):
        if tool == 'list_endpoints':
            return json.dumps(_listing(STREAMLAKE))
        raise RuntimeError(f"MCP tool 'chat' at {url} timed out after {timeout_seconds}s")

    monkeypatch.setattr(so, 'call_mcp_tool', fake_call)
    with pytest.raises(RuntimeError, match='timed out'):
        await so.ScrapeSession(_no_browser, 'q', provider_pin=pin).run('sys', 'user', [])


async def test_an_unpinned_run_sends_no_provider(monkeypatch):
    sent = []
    _wire(monkeypatch, _listing(), [_chat()], sent)
    await so.ScrapeSession(_no_browser, 'q', provider_pin=so.ProviderPin(MODEL)).run('sys', 'user', [])
    assert 'provider' not in sent[0][1]


# --- cost alert ------------------------------------------------------------------------------------

async def test_a_query_far_above_the_estimate_raises_a_run_alert(monkeypatch, caplog):
    pin = await _pinned(monkeypatch, _listing(STREAMLAKE))
    monkeypatch.setattr(tools, '_ui_alerts', [])
    monkeypatch.setattr(tools, '_current_query', 'AI Architect')
    prompt = 1_000_000
    too_dear = pin.estimate_per_million * (SCRAPER_COST_ALERT_FACTOR + 1)
    _wire(monkeypatch, _listing(STREAMLAKE), [_chat(cost=too_dear, prompt=prompt, cached=900_000)])
    stats = {'cost': 0.0, 'input_tokens': 0, 'output_tokens': 0, 'cache_read_input_tokens': 0}
    run_pass = agent._openrouter_run_pass(_no_browser, [], stats, {}, pin)
    with caplog.at_level(logging.INFO):
        await run_pass('instruction')
    assert [a['kind'] for a in tools._ui_alerts] == ['scrape_cost']
    assert any('served by StreamLake' in r.message and '/M prompt' in r.message for r in caplog.records)
    assert any('SCRAPER COST HIGH' in line for line in agent.assess_run_health({'ui_alerts': tools._ui_alerts}))


async def test_a_query_at_the_estimate_raises_nothing(monkeypatch):
    pin = await _pinned(monkeypatch, _listing(STREAMLAKE))
    monkeypatch.setattr(tools, '_ui_alerts', [])
    monkeypatch.setattr(tools, '_current_query', 'AI Architect')
    _wire(monkeypatch, _listing(STREAMLAKE), [_chat(cost=pin.estimate_per_million, prompt=1_000_000, cached=940_000)])
    stats = {'cost': 0.0, 'input_tokens': 0, 'output_tokens': 0, 'cache_read_input_tokens': 0}
    await agent._openrouter_run_pass(_no_browser, [], stats, {}, pin)('instruction')
    assert tools._ui_alerts == []
