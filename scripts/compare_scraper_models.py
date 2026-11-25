'''Run the same Stage 1b searches with several scraper models and compare what each found.

Why this is a tracked script
----------------------------
Stage 1b is the costliest stage of a run (about $0.23 of $0.32 over the runs of 2026-11-13..24), so
the question is whether a free local model can drive the search page as well. Each model drives the
real search page through the production loop: `agent.run_stage_1b` with the model as
`config.SCRAPER_MODEL_OVERRIDE`. That keeps the guards, the code-side harvest, the region checks,
the recovery pass and the pacing. A model's listings are only those its own loop recorded: the
Anthropic fallback is off (`fallback=False`), so a model that fails is reported as failed.

Nothing a real run reads is changed. `tools_generic._replay_dir` keeps processed_jobs/ untouched,
so a listing seen here is still new to the next run. No Stage 2 runs and nothing goes to Telegram.

This DRIVES THE LINKEDIN ACCOUNT: one Stage 1b pass per model over the chosen queries, paced as a
run is. The run lock is taken, so it cannot overlap a real run. AGENTS.md: start it only with the
user's go-ahead.

Usage:
    uv run python scripts/compare_scraper_models.py                       # default models, first generated query
    uv run python scripts/compare_scraper_models.py --query "Staff AI Engineer" --model gpt-oss:20b
    uv run python scripts/compare_scraper_models.py --searches 2 --browser minimized

Output: run_dir/replays/scraper-compare-<timestamp>/, holding one <model>.json per model
(listings, per-query stats, alerts, error) and report.md. The first model is the baseline that the
others' listings are compared with.
'''

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import agentic_job_search.config as config
from agentic_job_search import agent, system_sleep
from agentic_job_search import tools_generic as tools_module
from agentic_job_search.config import SCRAPER_INTER_QUERY_DELAY_SECONDS, SECONDS_PER_MINUTE
from agentic_job_search.linkedin_session import check_linkedin_session
from agentic_job_search.preferences import RUN_DIR
from agentic_job_search.snapshot_prune import SNAPSHOT_PRUNE_LEVELS
from agentic_job_search.triage import list_local_models
from utils_tools_n_agents_common.logging_setup import setup_logging
from utils_tools_n_agents_common.mcp_client import unwrap_exception
from utils_tools_n_agents_common.models import OPENROUTER_MODEL_NAME_SCRAPER

logger = logging.getLogger('compare_scraper_models')

# Installed Ollama models whose tool calling tools_llm_local's test_model_capabilities proved,
# chosen 2026-11-24 for speed of prompt processing (the cost of a long scraper conversation) and
# agentic score: gpt-oss:20b is the fastest at prefill, Nex-N2.5-mini scores highest on agentic work.
DEFAULT_MODELS = [
    OPENROUTER_MODEL_NAME_SCRAPER,
    'gpt-oss:20b',
    'qwen3.6:35b-mlx',
    'hf.co/bartowski/nex-agi_Nex-N2.5-mini-GGUF:Q4_K_M',
]
REPORT_FILENAME = 'report.md'
RUN_LOCK_MODE = 'compare-scraper-models'


def model_filename(model: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', model) + '.json'


async def check_models(models: list[str]) -> list[str]:
    '''Every reason a model cannot run, before the browser opens. Local models need the server up.'''
    problems = []
    if any(agent.route_for(model) == 'ollama' for model in models):
        try:
            await list_local_models()
        except Exception as ex:
            problems.append(f'tools_llm_local is unreachable ({config.LLM_LOCAL_MCP_URL}): {unwrap_exception(ex)}')
            return problems
    for model in models:
        if problem := await agent.scraper_model_preflight(model):
            problems.append(f'{model}: {problem}')
    return problems


async def scrape_with(model: str, queries: list[str], browser: 'agent.BrowserServer', out_dir: Path) -> dict:
    '''One Stage 1b pass over `queries` with `model`. Its failure is returned, never raised.'''
    agent._reset_run_state()
    tools_module._replay_dir = out_dir
    tools_module._scrape_per_query = {}
    tools_module.load_processed_jobs()
    config.SCRAPER_MODEL_OVERRIDE = model
    stats = agent.new_stage_stats()
    error = None
    started = time.monotonic()
    logger.info(f'Comparison: {model} on {len(queries)} quer(ies)')
    try:
        await agent.run_stage_1b(browser.port, browser.mcp, queries, stats, browser=browser, fallback=False)
    except Exception as ex:
        error = f'{type(ex).__name__}: {unwrap_exception(ex)}'
        logger.error(f'Comparison: {model} failed after {time.monotonic() - started:.0f}s: {error}')
    finally:
        config.SCRAPER_MODEL_OVERRIDE = None
    listings = [{'job_id': job_id, 'site': site, 'title': record.get('title'), 'company': record.get('company'),
                 'query': record.get('query'), 'region': record.get('region'),
                 'check_status': record.get('check_status')}
                for (site, job_id), record in tools_module._listing_records.items()]
    return {
        'model': model, 'queries': queries, 'seconds': round(time.monotonic() - started, 1), 'error': error,
        'cost_usd': stats['cost'], 'stage_stats': stats, 'per_query': dict(tools_module._scrape_per_query),
        'queries_searched': {q: str(v) for q, v in tools_module._queries_searched.items()},
        'query_errors': {q: str(v) for q, v in tools_module._query_errors.items()},
        'alerts': list(tools_module._ui_alerts), 'listings': listings,
    }


# A table cell holds the start of an error; the model's own section below gives it whole.
ERROR_CELL_MAX_CHARS = 120


def error_summary(result: dict) -> str:
    '''The run's error, else its queries' errors (run_scraper keeps a query's failure there), for one table cell.'''
    text = result['error'] or '; '.join(result['query_errors'].values())
    text = ' '.join(text.split()).replace('|', '/')
    return text if len(text) <= ERROR_CELL_MAX_CHARS else f'{text[:ERROR_CELL_MAX_CHARS]}… (cut; whole text below)'


def report(results: list[dict]) -> str:
    '''The comparison as Markdown: one row per model, then what each found apart from the baseline.'''
    baseline = results[0]
    baseline_ids = {item['job_id'] for item in baseline['listings']}
    lines = [f'# Stage 1b scraper comparison, {datetime.now():%Y-%m-%d %H:%M}', '',
             f'Queries: {", ".join(repr(q) for q in baseline["queries"])}. Baseline: `{baseline["model"]}`.', '',
             '| Model | Listings | Shared with baseline | Only this model | Turns | Model time (min) | '
             'Wall time (min) | Cost (USD) | Page results elided | Alerts | Error |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for result in results:
        ids = {item['job_id'] for item in result['listings']}
        turns = sum(entry.get('iterations', 0) for entry in result['per_query'].values())
        model_seconds = sum(entry.get('model_seconds', 0.0) for entry in result['per_query'].values())
        elided = sum(entry.get('elided_results', 0) for entry in result['per_query'].values())
        lines.append(
            f'| `{result["model"]}` | {len(ids)} | {len(ids & baseline_ids)} | {len(ids - baseline_ids)} | {turns} | '
            f'{model_seconds / SECONDS_PER_MINUTE:.1f} | {result["seconds"] / SECONDS_PER_MINUTE:.1f} | {result["cost_usd"]:.4f} | '
            f'{elided} | {len(result["alerts"])} | {error_summary(result)} |')
    for result in results:
        lines += ['', f'## `{result["model"]}`', '']
        for query, outcome in result['queries_searched'].items():
            lines.append(f'- searched {query!r}: {outcome}')
        for query, failure in result['query_errors'].items():
            lines.append(f'- **error** in {query!r}: {failure}')
        for alert in result['alerts']:
            lines.append(f'- alert `{alert.get("kind")}` ({alert.get("query")} / {alert.get("region")}): '
                         f'{alert.get("detail")}')
        if result is not baseline:
            ids = {item['job_id'] for item in result['listings']}
            missed = [item for item in baseline['listings'] if item['job_id'] not in ids]
            extra = [item for item in result['listings'] if item['job_id'] not in baseline_ids]
            for label, items in (('Baseline found, this model did not', missed), ('Only this model found', extra)):
                lines += ['', f'{label} ({len(items)}):', '']
                lines += [f'- {item["job_id"]} {item["title"]} — {item["company"]} ({item["region"]})' for item in items]
    return '\n'.join(lines) + '\n'


async def main() -> int:
    setup_logging('compare-scraper-models', RUN_DIR, level=os.environ.get('LOG_LEVEL', 'INFO'))
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--model', action='append', help='A scraper model, repeatable; the first is the baseline. '
                        'Default: DEFAULT_MODELS.')
    parser.add_argument('--query', action='append', help='A search query, repeatable. Default: the first '
                        '--searches queries Stage 1a generates.')
    parser.add_argument('--searches', type=int, default=1, help='Generated queries to use when no --query is given.')
    parser.add_argument('--browser', choices=['headless', 'minimized', 'visible'], default='headless')
    parser.add_argument('--scraper-snapshot-prune', choices=['off', *SNAPSHOT_PRUNE_LEVELS], default=None,
                        help='What every model is shown of a snapshot (default SCRAPER_SNAPSHOT_PRUNE_LEVEL).')
    args = parser.parse_args()
    models = args.model or DEFAULT_MODELS
    if args.scraper_snapshot_prune:
        config.SCRAPER_SNAPSHOT_PRUNE_LEVEL = None if args.scraper_snapshot_prune == 'off' else args.scraper_snapshot_prune
    logger.info(f'Snapshot prune level: {config.SCRAPER_SNAPSHOT_PRUNE_LEVEL or "off"}')

    if problems := await check_models(models):
        for problem in problems:
            logger.error(f'Comparison not started: {problem}')
        return 1
    if conflict := tools_module.acquire_run_lock(mode=RUN_LOCK_MODE):
        logger.error(f"Comparison not started: a run is active (pid {conflict['pid']}, started "
                     f"{conflict['started_at']}, {conflict['mode']}, command {conflict['argv']!r})")
        return 1
    out_dir = RUN_DIR / 'replays' / f'scraper-compare-{datetime.now():%Y%m%d-%H%M%S}'
    out_dir.mkdir(parents=True)
    awake = system_sleep.hold_awake()
    browser = agent.BrowserServer(args.browser)
    results: list[dict] = []
    try:
        tools_module.load_processed_jobs()
        await tools_module.load_applied_jobs()
        await tools_module.build_applied_organizations({})   # from the file only: no page is read here
        await browser.start()
        await agent.endpoint_pin.pin_single_turn_endpoints(agent.MODEL_NAME_SINGLE_TURN)
        async with agent.mcp_session(browser.mcp['url']) as call:
            check = await check_linkedin_session(call)
        if not check.signed_in:
            logger.error(f'Comparison not started: LinkedIn is signed out at {check.url} ({check.reason}). '
                         f'Sign in again: {agent.LINKEDIN_SIGN_IN_COMMAND}')
            return 1
        tools_module.set_card_rejection(agent.card_rejection_reason)
        queries = args.query or (await agent.generate_search_queries(agent.new_stage_stats()))[:args.searches]
        if not queries:
            logger.error('Comparison not started: no search queries')
            return 1
        logger.info(f'Comparison of {len(models)} model(s) on {queries} into {out_dir}')
        for position, model in enumerate(models):
            if position:
                pause = random.uniform(*SCRAPER_INTER_QUERY_DELAY_SECONDS)
                logger.info(f'Comparison: pausing {pause:.0f}s before the next model')
                await asyncio.sleep(pause)
            result = await scrape_with(model, queries, browser, out_dir)
            results.append(result)
            (out_dir / model_filename(model)).write_text(json.dumps(result, indent=tools_module.JSON_INDENT, default=str),
                                                         encoding='utf-8')
            # Written after every model, so a long comparison that stops still leaves a report.
            (out_dir / REPORT_FILENAME).write_text(report(results), encoding='utf-8')
            logger.info(f'Comparison: {model} recorded {len(result["listings"])} listing(s) in '
                        f'{result["seconds"] / SECONDS_PER_MINUTE:.1f} min; report {out_dir / REPORT_FILENAME}')
    finally:
        tools_module._replay_dir = None
        await browser.stop()
        system_sleep.release_awake(awake)
        tools_module.release_run_lock()
    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
