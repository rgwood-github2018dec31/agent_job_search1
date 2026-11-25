'''Ask each scraper model for single Stage 1b turns on saved pages, offline: a valid, safe tool call, and how fast?

Why this is a tracked script
----------------------------
A live comparison (`scripts/compare_scraper_models.py`) drives the LinkedIn account for every
model, and on 2026-11-24 all three local models failed there in ways one turn would have shown (a
tool call Ollama could not parse, a turn over the 300 s timeout). This asks the same question with
no account traffic, so only a model that passes it gets a live run.

Nothing here touches LinkedIn. The tool list comes from the pinned @playwright/mcp started with a
throwaway in-memory profile (`--isolated`), which never opens a page. The pages are search-page
snapshots a real run saved under `PLAYWRIGHT_OUTPUT_DIR`. Each model is asked through the
production path (`ScrapeSession._chat_local` for a local model: the same translation, context
check and parse retry), with the production system prompt, query instruction and tool list.

Scenarios, each one turn:
  start        system prompt and query instruction only: the first call.
  search_page  after browser_navigate and a browser_snapshot of a search page.
  late         after several snapshots, all but the newest elided as the loop's window does.

A turn passes when the reply is one or more tool calls that each name a served tool, carry
arguments that parse as an object with the schema's required fields, are not refused by the
scraper's guard, and name only refs present on the page the model was shown.

Usage:
    uv run python scripts/compare_scraper_models_offline.py                         # DEFAULT_MODELS
    uv run python scripts/compare_scraper_models_offline.py --model gpt-oss:20b --repeat 2

Output: run_dir/replays/scraper-offline-<timestamp>/results.json and report.md.
'''

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

from agentic_job_search import agent
from agentic_job_search import scrape_openrouter as so
from agentic_job_search import tools_generic as tools_module
from agentic_job_search.config import (
    SCRAPER_LOCAL_PAGE_RESULTS_KEPT,
    SCRAPER_SNAPSHOT_PRUNE_LEVEL,
    SCRAPER_SNAPSHOT_RESULT_MAX_CHARS,
)
from agentic_job_search.preferences import RUN_DIR
from agentic_job_search.snapshot_prune import SNAPSHOT_PRUNE_LEVELS, prune_search_snapshot
from agentic_job_search.text_budget import snippet, truncate_reported
from utils_tools_n_agents_common.logging_setup import setup_logging
from utils_tools_n_agents_common.mcp_client import unwrap_exception

logger = logging.getLogger('compare_scraper_models_offline')

# Installed Ollama models whose tool calling tools_llm_local's test_model_capabilities proved and
# whose context reaches SCRAPER_LOCAL_NUM_CTX, chosen 2026-11-25. Sizes are of the installed tag.
DEFAULT_MODELS = [
    'gpt-oss:20b',                                          # MoE 20B / 3.6B active, MXFP4
    'qwen3.6:35b-mlx',                                      # Ollama MLX, MoE 35B / 3B active, nvfp4
    'hf.co/bartowski/nex-agi_Nex-N2.5-mini-GGUF:Q4_K_M',    # MoE, 3B active, Q4_K_M
    'hf.co/poolside/Laguna-XS-2.1-GGUF:Q4_K_M',             # MoE, 3B active, Q4_K_M; agentic coding
    'hf.co/LiquidAI/LFM2.5-24B-A2B-GGUF:Q4_K_M',            # MoE 24B / 2B active, Q4_K_M
    'qwen3.5:9b-q8_0',                                      # dense 9B, Q8_0
    'ministral-3:latest',                                   # dense, Q4_K_M
    'qwen3.6:27b-mlx',                                      # Ollama MLX, dense 27B, nvfp4
]
QUERY = 'Staff AI Engineer'
# The search URL the system prompt prescribes (agent.build_scraper_prompt): the query plus ', remote',
# and an empty f_SAL. A scripted navigation elsewhere made models navigate again (2026-11-25).
SEARCH_URL = f'https://www.linkedin.com/jobs/search-results/?keywords={quote_plus(QUERY + ", remote")}&f_SAL='
# Filters the prompt says to set by clicking chips, never in the URL.
URL_FILTER_PARAMETER_RE = re.compile(r'[?&](geoId|f_TPR|f_E|location)=')
# Saved snapshots of the search page (results, filter chips, detail pane) from the baseline query
# of 2026-11-24 (deepseek/deepseek-v4.1-flash-1017, 18:59-19:07 local), by file name.
SEARCH_PAGE_SNAPSHOT = 'page-2026-11-25T00-00-08-305Z.yml'
LATE_SNAPSHOTS = ['page-2026-11-25T00-00-08-305Z.yml', 'page-2026-11-25T00-00-27-648Z.yml',
                  'page-2026-11-25T00-00-37-588Z.yml', 'page-2026-11-25T00-02-43-829Z.yml']
# The clicks between the late snapshots: the text of the element clicked on the snapshot before each.
# Their refs are looked up on that page: made-up refs were copied by a model (Nex, 2026-11-25).
LATE_CLICKS = ['Filter by Date posted', 'Filter by Date posted', 'Past week']
PAGE_REF_RE = re.compile(r'\[ref=([A-Za-z0-9]+)\]')
REF_VALUE_RE = re.compile(r'(e\d+|f\d+e\d+)')
REF_ARGUMENT_KEYS = ('ref', 'target')
# A CSS selector carries quotes only inside an attribute test ([aria-label="..."]); a target with
# quotes outside one is an accessible name or a locator, which @playwright/mcp rejects.
CSS_ATTRIBUTE_TEST_RE = re.compile(r'\[[^\]]*\]')
REPORT_FILENAME = 'report.md'
RESULTS_FILENAME = 'results.json'


def snapshot_result(snapshot_dir: Path, name: str, query: str, prune_level: str | None = None) -> str:
    '''A browser_snapshot result as @playwright/mcp 0.0.86 words it, processed as the loop does.'''
    path = snapshot_dir / name
    try:
        page = path.read_text(encoding='utf-8')
    except OSError as ex:
        raise RuntimeError(f'cannot read saved snapshot {path} ({type(ex).__name__}: {ex}); a real run '
                           f'writes them to agent.PLAYWRIGHT_OUTPUT_DIR, which is cleared on reboot') from ex
    text = f'### Page\n- Page URL: {SEARCH_URL}\n### Snapshot\n```yaml\n{page}\n```'
    text = so.drop_job_detail_pane(text, f'saved snapshot {name}')
    if prune_level:
        text = prune_search_snapshot(text, prune_level, f'saved snapshot {name}')
    return truncate_reported(text, SCRAPER_SNAPSHOT_RESULT_MAX_CHARS, f'saved snapshot {name}')


def action_result(code: str, snapshot_name: str) -> str:
    '''An action's result: @playwright/mcp links the page snapshot instead of inlining it.'''
    return (f'### Ran Playwright code\n```js\n{code}\n```\n### Page\n- Page URL: {SEARCH_URL}\n'
            f'### Snapshot\n- [Snapshot]({snapshot_name})')


def clicked_ref(path: Path, label: str) -> str:
    '''The ref of the first node on a saved page whose line holds `label`; raises when there is none.'''
    for line in path.read_text(encoding='utf-8').splitlines():
        if label in line and (match := PAGE_REF_RE.search(line)):
            return match.group(1)
    raise RuntimeError(f'no node with a ref holds {label!r} in saved snapshot {path}')


def scripted_call(call_id: str, name: str, args: dict) -> dict:
    return {'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}


def scenarios(system_prompt: str, instruction: str, snapshot_dir: Path,
              prune_level: str | None = None) -> dict[str, tuple[list[dict], str]]:
    '''Each scenario's conversation, and the page text whose refs its next call may name.'''
    opening = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': instruction}]
    search_page = snapshot_result(snapshot_dir, SEARCH_PAGE_SNAPSHOT, QUERY, prune_level)
    navigated = [*opening, scripted_call('nav', 'browser_navigate', {'url': SEARCH_URL}),
                 {'role': 'tool', 'tool_call_id': 'nav',
                  'content': action_result(f"await page.goto('{SEARCH_URL}');", SEARCH_PAGE_SNAPSHOT)},
                 scripted_call('snap0', 'browser_snapshot', {}),
                 {'role': 'tool', 'tool_call_id': 'snap0', 'content': search_page}]
    late = list(navigated[:-2])
    for n, name in enumerate(LATE_SNAPSHOTS):
        late += [scripted_call(f'snap{n}', 'browser_snapshot', {}),
                 {'role': 'tool', 'tool_call_id': f'snap{n}', 'content': snapshot_result(snapshot_dir, name, QUERY, prune_level)}]
        if n < len(LATE_CLICKS):
            ref = clicked_ref(snapshot_dir / name, LATE_CLICKS[n])
            late += [scripted_call(f'click{n}', 'browser_click', {'element': LATE_CLICKS[n], 'target': ref}),
                     {'role': 'tool', 'tool_call_id': f'click{n}',
                      'content': action_result(f"await page.getByRole('button', {{ name: '{LATE_CLICKS[n]}' }}).click();",
                                               LATE_SNAPSHOTS[n + 1])}]
    so.elide_superseded_page_results(late, SCRAPER_LOCAL_PAGE_RESULTS_KEPT or 1)
    return {'start': (opening, ''), 'search_page': (navigated, search_page),
            'late': (late, late[-1]['content'])}


def judge(reply: dict, tools: list[dict], page: str) -> list[str]:
    '''Every reason this reply is not a usable next step; empty when it is.'''
    calls = reply.get('tool_calls') or []
    if not calls:
        return [f'no tool call (finish_reason={reply.get("finish_reason")}): {snippet(reply.get("content") or "")!r}']
    schemas = {t['function']['name']: t['function'].get('parameters') or {} for t in tools}
    page_refs = set(PAGE_REF_RE.findall(page))
    problems = []
    for call in calls:
        name = call['function']['name']
        if name not in schemas:
            problems.append(f'{name}: not a served tool')
            continue
        try:
            args = json.loads(call['function'].get('arguments') or '{}')
        except json.JSONDecodeError as ex:
            problems.append(f'{name}: arguments are not JSON ({ex}): {snippet(call["function"].get("arguments"))!r}')
            continue
        if not isinstance(args, dict):
            problems.append(f'{name}: arguments are a {type(args).__name__}, not an object')
            continue
        if missing := [k for k in schemas[name].get('required', []) if k not in args]:
            problems.append(f'{name}: missing required {missing}')
        if name == 'browser_navigate' and URL_FILTER_PARAMETER_RE.search(str(args.get('url', ''))):
            problems.append(f'{name}: sets a filter in the URL, which the prompt forbids: {args.get("url")!r}')
        if (refusal := so._guard_call(name, args)) is not None:
            problems.append(f'{name}: refused by the guard: {snippet(refusal)}')
        for key in REF_ARGUMENT_KEYS:
            value = str(args.get(key, ''))
            if REF_VALUE_RE.fullmatch(value) and value not in page_refs:
                problems.append(f'{name}: {key}={value!r} is not on the page it was shown')
            elif value and not REF_VALUE_RE.fullmatch(value) and (
                    value.startswith('ref=') or re.search('["\']', CSS_ATTRIBUTE_TEST_RE.sub('', value))):
                problems.append(f'{name}: {key}={value!r} is neither a ref nor a CSS selector')
    return problems


async def browser_tool_defs() -> list[dict]:
    '''The scraper's tool list, from the pinned @playwright/mcp on a throwaway profile.'''
    with tempfile.TemporaryDirectory(prefix='scraper-offline-pw-') as output_dir:
        port = agent.find_free_port()
        proc = await agent.start_playwright_server(port, browser_mode='headless', isolated_output_dir=Path(output_dir))
        try:
            async with agent.mcp_session(f'http://localhost:{port}/mcp') as call:
                return so.browser_tool_defs(await call.list_tools()) + so.LOCAL_TOOL_DEFS
        finally:
            proc.terminate()
            await proc.wait()


async def ask(model: str, messages: list[dict], tools: list[dict]) -> dict:
    session = so.ScrapeSession(lambda *a: None, QUERY, model=model)
    started = time.monotonic()
    reply = json.loads(await session._chat_local({'messages': messages, 'tools': tools}))
    reply['seconds'] = round(time.monotonic() - started, 1)
    return reply


def report(results: list[dict]) -> str:
    lines = [f'# Stage 1b scraper models, one turn on saved pages, {datetime.now():%Y-%m-%d %H:%M}', '',
             'Pass: a tool call with valid arguments the guard allows, naming only refs on the page shown.', '',
             '| Model | Scenario | Pass | Seconds | Prompt tokens (reused) | Call | Problem |',
             '|---|---|---|---|---|---|---|']
    for r in results:
        call = '; '.join(f'{c["function"]["name"]} {snippet(c["function"].get("arguments"), 80)}'
                         for c in r.get('tool_calls') or [])
        usage = r.get('usage') or {}
        tokens = (f'{usage.get("prompt_tokens", 0):,} ({(usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0):,})'
                  if usage else '')
        problem = '; '.join(r['problems']).replace('|', '/')
        lines.append(f'| `{r["model"]}` | {r["scenario"]} | {"yes" if not r["problems"] else "**no**"} | '
                     f'{r.get("seconds", "")} | {tokens} | {call.replace("|", "/")} | {problem} |')
    return '\n'.join(lines) + '\n'


async def rejudge(run_dir: Path, snapshot_dir: Path, prune_level: str | None) -> int:
    '''Score a finished run's saved replies again with the current judge, rewriting its report.

    Pass the run's own --snapshot-prune-level (its directory name ends with it): refs are checked on
    the pages as that run showed them.
    '''
    results = json.loads((run_dir / RESULTS_FILENAME).read_text(encoding='utf-8'))
    tools = await browser_tool_defs()
    cases = scenarios(agent.build_scraper_prompt(), agent.scraper_query_instruction(QUERY), snapshot_dir, prune_level)
    for result in results:
        if result.get('tool_calls') is not None or 'content' in result:   # a reply came back, not an exception
            result['problems'] = judge(result, tools, cases[result['scenario']][1])
    (run_dir / RESULTS_FILENAME).write_text(json.dumps(results, indent=tools_module.JSON_INDENT, default=str),
                                            encoding='utf-8')
    (run_dir / REPORT_FILENAME).write_text(report(results), encoding='utf-8')
    logger.info(f'Re-scored {len(results)} turn(s): {run_dir / REPORT_FILENAME}')
    return 0


async def main() -> int:
    setup_logging('compare-scraper-models-offline', RUN_DIR, level=os.environ.get('LOG_LEVEL', 'INFO'))
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--model', action='append', help='A local scraper model, repeatable. Default: DEFAULT_MODELS.')
    parser.add_argument('--repeat', type=int, default=1, help='Times each scenario is asked of each model.')
    parser.add_argument('--snapshot-dir', type=Path, default=agent.PLAYWRIGHT_OUTPUT_DIR)
    parser.add_argument('--snapshot-prune-level', choices=['off', *SNAPSHOT_PRUNE_LEVELS],
                        default=SCRAPER_SNAPSHOT_PRUNE_LEVEL or 'off',
                        help='Prune each saved page as Stage 1b would (default SCRAPER_SNAPSHOT_PRUNE_LEVEL).')
    parser.add_argument('--rejudge', type=Path, help='Re-score a finished run directory with the current '
                        'checks (no model is asked; the tool list is read again).')
    args = parser.parse_args()
    if args.snapshot_prune_level == 'off':
        args.snapshot_prune_level = None
    if args.rejudge:
        return await rejudge(args.rejudge, args.snapshot_dir, args.snapshot_prune_level)
    models = args.model or DEFAULT_MODELS
    for model in models:
        if problem := await agent.scraper_model_preflight(model):
            logger.error(f'Not started: {model}: {problem}')
            return 1
    tools_module._current_query = QUERY
    tools = await browser_tool_defs()
    cases = scenarios(agent.build_scraper_prompt(), agent.scraper_query_instruction(QUERY), args.snapshot_dir,
                      args.snapshot_prune_level)
    out_dir = RUN_DIR / 'replays' / f'scraper-offline-{datetime.now():%Y%m%d-%H%M%S}-{args.snapshot_prune_level or "unpruned"}'
    out_dir.mkdir(parents=True)
    logger.info(f'{len(models)} model(s) x {len(cases)} scenario(s) x {args.repeat}, {len(tools)} tools, snapshot '
                f'prune level {args.snapshot_prune_level or "none"}, into {out_dir}')
    results: list[dict] = []
    for model in models:
        for scenario, (messages, page) in cases.items():
            for attempt in range(args.repeat):
                try:
                    reply = await ask(model, messages, tools)
                    problems = judge(reply, tools, page)
                except Exception as ex:
                    reply, problems = {}, [f'{type(ex).__name__}: {snippet(unwrap_exception(ex), 400)}']
                results.append({'model': model, 'scenario': scenario, 'attempt': attempt, **reply, 'problems': problems})
                logger.info(f'{model} / {scenario} #{attempt + 1}: {"pass" if not problems else "FAIL"} '
                            f'in {reply.get("seconds", "?")}s'
                            + (f' — {"; ".join(problems)}' if problems else
                               f' — {", ".join(c["function"]["name"] for c in reply.get("tool_calls") or [])}'))
                (out_dir / RESULTS_FILENAME).write_text(json.dumps(results, indent=tools_module.JSON_INDENT, default=str),
                                                        encoding='utf-8')
                (out_dir / REPORT_FILENAME).write_text(report(results), encoding='utf-8')
    logger.info(f'Report: {out_dir / REPORT_FILENAME}')
    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
