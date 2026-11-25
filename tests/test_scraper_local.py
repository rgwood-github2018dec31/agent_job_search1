'''The local route of the Stage 1b function-calling loop, and the comparison's no-record guard.

A local scraper model (an Ollama tag) runs the same loop as the OpenRouter one, through
tools_llm_local's chat. What can go wrong is the translation between the two conversation shapes,
a prompt Ollama would cut from the front without saying so, and a local name quietly ending up on
the Anthropic scraper. Each test below fails if one of those comes back.
'''
import json
import math
from datetime import date

import pytest
from agentic_job_search import agent
from agentic_job_search import scrape_openrouter as so
from agentic_job_search import tools_generic as tools
from agentic_job_search.config import (
    LLM_LOCAL_MCP_URL,
    LLM_LOCAL_PURPOSE_SCRAPER,
    LLM_LOCAL_SERVICE_NAME,
    SCRAPER_LOCAL_CHARS_PER_TOKEN,
    SCRAPER_LOCAL_NUM_CTX,
)

LOCAL_MODEL = 'gpt-oss:20b'
TOOLS = [{'type': 'function', 'function': {'name': 'harvest_listings', 'description': 'd',
                                           'parameters': {'type': 'object', 'properties': {}}}}]


def test_ollama_messages_turns_argument_strings_into_objects_and_names_tool_results():
    messages = [
        {'role': 'system', 'content': 'sys'},
        {'role': 'user', 'content': 'go'},
        {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'call_a', 'type': 'function', 'function': {'name': 'browser_click', 'arguments': '{"ref": "e1"}'}},
            {'id': 'call_b', 'type': 'function', 'function': {'name': 'harvest_listings', 'arguments': 'not json'}},
        ]},
        {'role': 'tool', 'tool_call_id': 'call_a', 'content': 'clicked'},
        {'role': 'tool', 'tool_call_id': 'call_b', 'content': 'ERROR: invalid JSON'},
    ]
    out = so.ollama_messages(messages)
    assert out[0] == {'role': 'system', 'content': 'sys'}
    assert out[2]['tool_calls'][0] == {'function': {'name': 'browser_click', 'arguments': {'ref': 'e1'}}}
    assert out[2]['tool_calls'][1]['function']['arguments'] == {'_unparsed_arguments': 'not json'}
    assert out[3] == {'role': 'tool', 'content': 'clicked', 'tool_name': 'browser_click'}
    assert out[4]['tool_name'] == 'harvest_listings'


def test_local_reply_gets_call_ids_and_json_string_arguments():
    raw = json.dumps({'model': LOCAL_MODEL, 'done_reason': 'stop', 'prompt_eval_count': 900, 'eval_count': 12,
                      'message': {'role': 'assistant', 'content': '', 'tool_calls': [
                          {'function': {'name': 'harvest_listings', 'arguments': {}}},
                          {'function': {'name': 'browser_click', 'arguments': {'ref': 'e9'}}}]}})
    reply = so.local_reply_as_chat_reply(raw, LOCAL_MODEL, 'Q', 'call_7_')
    assert reply['ok'] is True and reply['cost_usd'] == 0.0
    assert [c['id'] for c in reply['tool_calls']] == ['call_7_0', 'call_7_1']
    assert json.loads(reply['tool_calls'][1]['function']['arguments']) == {'ref': 'e9'}
    assert reply['usage'] == {'prompt_tokens': 900, 'completion_tokens': 12,
                              'prompt_tokens_details': {'cached_tokens': 0}}
    assert reply['content'] is None


def test_local_reply_without_tool_calls_reads_as_a_stop():
    raw = json.dumps({'done_reason': 'stop', 'message': {'role': 'assistant', 'content': 'done'}})
    reply = so.local_reply_as_chat_reply(raw, LOCAL_MODEL, 'Q', 'call_1_')
    assert reply['tool_calls'] is None and reply['content'] == 'done' and reply['finish_reason'] == 'stop'


@pytest.mark.parametrize('raw, match', [
    ('not json', 'is not JSON'),
    ('[1]', 'is not an object'),
    (json.dumps({'error': 'model not found'}), 'reported an error'),
    (json.dumps({'done': True}), 'has no message object'),
    (json.dumps({'message': {'tool_calls': [{'function': {'arguments': {}}}]}}), 'no function name'),
])
def test_unusable_local_reply_raises_with_the_body(raw, match):
    with pytest.raises((ValueError, RuntimeError), match=match) as info:
        so.local_reply_as_chat_reply(raw, LOCAL_MODEL, 'Q', 'call_1_')
    assert LOCAL_MODEL in str(info.value)


async def test_local_session_calls_the_local_server_with_tools_and_context(monkeypatch):
    sent = []

    async def fake_call(url, tool, args, timeout_seconds=None):
        sent.append((url, tool, args))
        return json.dumps({'done_reason': 'stop', 'message': {'role': 'assistant', 'content': 'nothing to do'}})

    monkeypatch.setattr(so, 'call_mcp_tool', fake_call)
    session = so.ScrapeSession(lambda *a: None, 'Q', model=LOCAL_MODEL)
    session.nudges = so.SCRAPER_NO_TOOL_NUDGES   # no nudge: one turn, then stop
    await session.run('sys', 'user', TOOLS)
    url, tool, args = sent[0]
    assert (url, tool) == (LLM_LOCAL_MCP_URL, 'chat')
    assert args['tools'] == TOOLS and args['num_ctx'] == SCRAPER_LOCAL_NUM_CTX
    assert args['service_name'] == LLM_LOCAL_SERVICE_NAME and args['purpose_slug'] == LLM_LOCAL_PURPOSE_SCRAPER
    assert 'provider' not in args
    assert session.cost == 0.0 and session.model_seconds >= 0.0


async def test_local_prompt_over_the_context_window_stops_before_sending(monkeypatch):
    async def must_not_call(*a, **k):
        raise AssertionError('an oversized prompt must not reach Ollama, which would cut it silently')

    monkeypatch.setattr(so, 'call_mcp_tool', must_not_call)
    session = so.ScrapeSession(lambda *a: None, 'Q', model=LOCAL_MODEL)
    huge = 'x' * int(SCRAPER_LOCAL_NUM_CTX * SCRAPER_LOCAL_CHARS_PER_TOKEN + 1)
    with pytest.raises(so.ScraperContextOverflowError, match='SCRAPER_LOCAL_NUM_CTX'):
        await session.run('sys', huge, TOOLS)


@pytest.mark.parametrize('model, ok', [
    (LOCAL_MODEL, True),
    ('hf.co/bartowski/nex-agi_Nex-N2.5-mini-GGUF:Q4_K_M', True),
    ('deepseek/deepseek-v4.1-flash-1017', True),
    ('mlx-lm-server:org/repo', False),
    ('claude-haiku-5-5', False),
])
def test_scraper_model_problem(model, ok):
    assert (so.scraper_model_problem(model) is None) is ok


async def test_a_local_scraper_model_runs_the_loop_not_the_anthropic_scraper(monkeypatch):
    monkeypatch.setattr(tools, '_ui_alerts', [])
    monkeypatch.setattr(tools, '_scrape_per_query', {})
    monkeypatch.setattr(agent, 'MODEL_NAME_SCRAPER', LOCAL_MODEL)

    class _Ok:
        async def __aenter__(self):
            async def call(*a, **k):
                return ''

            async def list_tools():
                return []
            call.list_tools = list_tools
            return call

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(agent, 'mcp_session', lambda url: _Ok())

    async def must_not_pin(model):
        raise AssertionError('a local model has no OpenRouter endpoint to pin')

    passes = []

    def fake_run_pass(browser_call, tool_defs, stats, per_query, provider_pin=None, model=None):
        passes.append((provider_pin, model))
        return lambda instruction: None

    async def fake_run_scraper(run_pass, queries, stats, before_query=None):
        return None

    async def fake_anthropic(playwright_mcp, queries, stats):
        raise AssertionError('a local scraper model must not become a Haiku run')

    monkeypatch.setattr(so, 'pin_scraper_provider', must_not_pin)
    monkeypatch.setattr(agent, '_openrouter_run_pass', fake_run_pass)
    monkeypatch.setattr(agent, 'run_scraper', fake_run_scraper)
    monkeypatch.setattr(agent, '_run_anthropic_scraper', fake_anthropic)
    assert await agent.run_stage_1b(1234, {'type': 'http'}, ['Q1'], {'cost': 0.0}) is True
    assert passes == [(None, LOCAL_MODEL)]


async def test_without_fallback_a_failure_is_raised_not_replaced(monkeypatch):
    monkeypatch.setattr(tools, '_ui_alerts', [])
    monkeypatch.setattr(agent, 'MODEL_NAME_SCRAPER', LOCAL_MODEL)

    class _Boom:
        async def __aenter__(self):
            raise RuntimeError('tools_llm_local down')

        async def __aexit__(self, *exc):
            return False

    async def fake_anthropic(playwright_mcp, queries, stats):
        raise AssertionError('fallback=False must not run the Anthropic scraper')

    monkeypatch.setattr(agent, 'mcp_session', lambda url: _Boom())
    monkeypatch.setattr(agent, '_run_anthropic_scraper', fake_anthropic)
    with pytest.raises(RuntimeError, match='tools_llm_local down'):
        await agent.run_stage_1b(1234, {'type': 'http'}, ['Q1'], {'cost': 0.0}, fallback=False)


def test_scraper_model_override_is_run_scoped(monkeypatch):
    import agentic_job_search.config as config
    monkeypatch.setattr(config, 'SCRAPER_MODEL_OVERRIDE', None)
    assert agent.scraper_model() == agent.MODEL_NAME_SCRAPER
    monkeypatch.setattr(config, 'SCRAPER_MODEL_OVERRIDE', LOCAL_MODEL)
    assert agent.scraper_model() == LOCAL_MODEL


async def test_a_replay_records_no_processed_job(tmp_path, monkeypatch):
    '''A comparison sees jobs no Stage 2 rates; recording them would dedup them away from the next run.'''
    monkeypatch.setattr(tools, 'RUN_DIR', tmp_path)
    monkeypatch.setattr(tools, 'PROCESSED_JOBS_DIR', tmp_path / 'processed_jobs')
    monkeypatch.setattr(tools, '_processed_jobs', set())
    monkeypatch.setattr(tools, '_replay_dir', tmp_path / 'replay')
    first = await tools.do_check_and_record_job('linkedin', '1234567890', 'Shopify', 'Senior Engineer',
                                                date_posted=date.today().isoformat())
    second = await tools.do_check_and_record_job('linkedin', '1234567890', 'Shopify', 'Senior Engineer',
                                                 date_posted=date.today().isoformat())
    assert first['content'][0]['text'] == 'new'
    assert second['content'][0]['text'] == 'already_processed'
    assert not (tmp_path / 'processed_jobs').exists()


def _page_conversation(sizes: list[int]) -> list[dict]:
    '''A conversation with one browser_snapshot result of each size, plus one long local tool result.'''
    messages = [{'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': 'go'}]
    for n, size in enumerate(sizes):
        messages.append({'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': f's{n}', 'type': 'function', 'function': {'name': 'browser_snapshot', 'arguments': '{}'}}]})
        messages.append({'role': 'tool', 'tool_call_id': f's{n}', 'content': f'- Page URL: page{n}\n' + 'x' * size})
    messages.append({'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': 'h', 'type': 'function', 'function': {'name': 'harvest_listings', 'arguments': '{}'}}]})
    messages.append({'role': 'tool', 'tool_call_id': 'h', 'content': 'y' * (so.SCRAPER_PAGE_RESULT_MIN_CHARS * 2)})
    return messages


def test_only_the_newest_page_results_stay_whole_and_each_stub_says_what_was_cut():
    big = so.SCRAPER_PAGE_RESULT_MIN_CHARS * 3
    messages = _page_conversation([big, big, so.SCRAPER_PAGE_RESULT_MIN_CHARS // 2, big])
    elided = so.elide_superseded_page_results(messages, keep=1)
    assert [(index, name) for index, name, _ in elided] == [(3, 'browser_snapshot'), (5, 'browser_snapshot')]
    stub = messages[3]['content']
    assert stub.startswith(so.ELIDED_RESULT_MARKER) and 'stale' in stub and 'page0' in stub
    assert f'{len("- Page URL: page0") + 1 + big:,} chars' in stub
    assert len(stub) < big
    assert messages[7]['content'].endswith('x')                 # short: the model's record of a step, kept
    assert messages[9]['content'].startswith('- Page URL: page3')   # the newest page read, whole
    assert messages[-1]['content'].startswith('y')              # a local tool's reply is never elided
    assert so.elide_superseded_page_results(messages, keep=1) == []   # a stub stays a stub: the prefix is stable


def test_the_window_applies_to_local_models_and_leaves_openrouter_unchanged():
    assert so.page_results_kept_for(LOCAL_MODEL) == so.SCRAPER_LOCAL_PAGE_RESULTS_KEPT
    assert so.page_results_kept_for('deepseek/deepseek-v4.1-flash-1017') == so.SCRAPER_OPENROUTER_PAGE_RESULTS_KEPT


def _local_reply(tool_name: str | None, prompt_eval_count: int = 900) -> str:
    calls = [{'function': {'name': tool_name, 'arguments': {}}}] if tool_name else []
    return json.dumps({'done_reason': 'stop', 'prompt_eval_count': prompt_eval_count, 'prompt_eval_cached_count': 800,
                       'eval_count': 5, 'message': {'role': 'assistant', 'content': '' if calls else 'done',
                                                    'tool_calls': calls}})


async def test_a_local_query_sends_each_old_snapshot_as_a_stub(monkeypatch):
    replies = iter([_local_reply('browser_snapshot'), _local_reply('browser_snapshot'), _local_reply(None)])
    sent = []

    async def fake_call(url, tool, args, timeout_seconds=None):
        sent.append(json.loads(json.dumps(args['messages'])))
        return next(replies)

    async def browser(name, args):
        return '- Page URL: search\n' + 'z' * (so.SCRAPER_PAGE_RESULT_MIN_CHARS * 2)

    monkeypatch.setattr(so, 'call_mcp_tool', fake_call)
    session = so.ScrapeSession(browser, 'Q', model=LOCAL_MODEL)
    session.nudges = so.SCRAPER_NO_TOOL_NUDGES
    await session.run('sys', 'user', TOOLS)
    tool_results = [m['content'] for m in sent[2] if m['role'] == 'tool']
    assert len(tool_results) == 2
    assert tool_results[0].startswith(so.ELIDED_RESULT_MARKER)
    assert tool_results[1].startswith('- Page URL: search')
    assert session.elided_results == 1 and session.usage['cached'] == 800 * 3


async def test_an_unparsable_tool_call_is_asked_again_without_changing_the_conversation(monkeypatch):
    sent = []

    async def fake_call(url, tool, args, timeout_seconds=None):
        sent.append(args['messages'])
        if len(sent) == 1:
            raise RuntimeError("POST http://localhost:11434/api/chat -> 500: {\"error\":\"error parsing tool call: "
                               "raw='We need to click'\"}")
        return _local_reply(None)

    monkeypatch.setattr(so, 'call_mcp_tool', fake_call)
    session = so.ScrapeSession(lambda *a: None, 'Q', model=LOCAL_MODEL)
    session.nudges = so.SCRAPER_NO_TOOL_NUDGES
    await session.run('sys', 'user', TOOLS)
    assert len(sent) == 2
    assert sent[1][:-1] == sent[0]
    assert sent[1][-1]['role'] == 'user' and 'could not be read as a tool call' in sent[1][-1]['content']


async def test_a_repeated_parse_failure_is_raised(monkeypatch):
    async def always_bad(url, tool, args, timeout_seconds=None):
        raise RuntimeError('500: error parsing tool call')

    monkeypatch.setattr(so, 'call_mcp_tool', always_bad)
    session = so.ScrapeSession(lambda *a: None, 'Q', model=LOCAL_MODEL)
    with pytest.raises(RuntimeError, match='error parsing tool call'):
        await session.run('sys', 'user', TOOLS)


async def test_a_reply_whose_prompt_filled_the_window_is_not_used(monkeypatch):
    full = math.ceil(SCRAPER_LOCAL_NUM_CTX * so.SCRAPER_LOCAL_CONTEXT_FULL_SHARE)

    async def fake_call(url, tool, args, timeout_seconds=None):
        return _local_reply('harvest_listings', prompt_eval_count=full)

    monkeypatch.setattr(so, 'call_mcp_tool', fake_call)
    session = so.ScrapeSession(lambda *a: None, 'Q', model=LOCAL_MODEL)
    with pytest.raises(so.ScraperContextOverflowError, match='may have cut it'):
        await session.run('sys', 'user', TOOLS)


def test_the_default_scraper_is_the_chosen_local_model_with_an_openrouter_fallback():
    from agentic_job_search import config
    assert config.MODEL_NAME_SCRAPER == 'qwen3.6:35b-mlx'
    assert so.scraper_model_problem(config.MODEL_NAME_SCRAPER) is None
    assert agent.route_for(config.SCRAPER_REMOTE_MODEL) == 'openrouter'


async def test_an_unavailable_local_scraper_hands_the_run_to_openrouter(monkeypatch):
    import agentic_job_search.config as config
    monkeypatch.setattr(config, 'SCRAPER_MODEL_OVERRIDE', None)
    monkeypatch.setattr(agent, 'MODEL_NAME_SCRAPER', LOCAL_MODEL)
    monkeypatch.setattr(tools, '_ui_alerts', [])

    async def not_installed(model):
        return f'{model} is not installed'

    monkeypatch.setattr(agent, 'scraper_model_preflight', not_installed)
    await agent.ensure_scraper_model_available()
    assert agent.scraper_model() == config.SCRAPER_REMOTE_MODEL
    assert [a['kind'] for a in tools._ui_alerts] == ['local_scraper_unavailable']


async def test_an_available_local_scraper_is_kept(monkeypatch):
    import agentic_job_search.config as config
    monkeypatch.setattr(config, 'SCRAPER_MODEL_OVERRIDE', None)
    monkeypatch.setattr(agent, 'MODEL_NAME_SCRAPER', LOCAL_MODEL)

    async def ok(model):
        return None

    monkeypatch.setattr(agent, 'scraper_model_preflight', ok)
    await agent.ensure_scraper_model_available()
    assert agent.scraper_model() == LOCAL_MODEL


async def test_queries_the_local_scraper_failed_are_searched_again_on_openrouter(monkeypatch):
    import agentic_job_search.config as config
    monkeypatch.setattr(tools, '_ui_alerts', [])
    monkeypatch.setattr(tools, '_queries_searched', {'Q1': 12, 'Q2': 'error'})
    monkeypatch.setattr(tools, '_query_errors', {'Q2': 'chat timed out'})
    pins, passes, searched = [], [], []

    async def fake_pin(model):
        pins.append(model)
        return 'pin'

    def fake_run_pass(browser_call, tool_defs, stats, per_query, provider_pin=None, model=None):
        passes.append((provider_pin, model))
        return lambda instruction: None

    async def fake_run_scraper(run_pass, queries, stats, before_query=None):
        searched.append(list(queries))

    monkeypatch.setattr(so, 'pin_scraper_provider', fake_pin)
    monkeypatch.setattr(agent, '_openrouter_run_pass', fake_run_pass)
    monkeypatch.setattr(agent, 'run_scraper', fake_run_scraper)
    await agent._rerun_failed_local_queries(LOCAL_MODEL, ['Q1', 'Q2'], None, TOOLS, {'cost': 0.0}, None)
    assert searched == [['Q2']]
    assert pins == [config.SCRAPER_REMOTE_MODEL] and passes == [('pin', config.SCRAPER_REMOTE_MODEL)]
    assert tools._ui_alerts[0]['kind'] == 'local_scraper_fallback' and 'chat timed out' in tools._ui_alerts[0]['detail']


async def test_nothing_is_searched_again_when_every_local_query_finished(monkeypatch):
    monkeypatch.setattr(tools, '_queries_searched', {'Q1': 12})

    async def must_not_run(*a, **k):
        raise AssertionError('no failed query, so no OpenRouter pass')

    monkeypatch.setattr(agent, 'run_scraper', must_not_run)
    await agent._rerun_failed_local_queries(LOCAL_MODEL, ['Q1'], None, TOOLS, {'cost': 0.0}, None)
