import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import socket
import sys
import tempfile
import time
from datetime import date, datetime
import requests
import yaml
from pathlib import Path
from typing import Any

from agentic_job_search.config import (
    EXTRACTOR_PROVIDER,
    JOB_STALE_AGE_DAYS,
    MAX_REFERENCE_JOBS,
    MODEL_NAME_LOW,
    MODEL_NAME_MEDIUM,
    RATING_PROVIDER,
    REFERENCE_SUMMARY_MAX_CHARS,
    THINKING_MAX_CHARS,
    TRIAGE_ENABLED,
)
from agentic_job_search.extract_openrouter import extract_job_page_openrouter
import agentic_job_search.tools_generic as tools_module
from agentic_job_search.tools_generic import (
    JOB_REQUIREMENTS_PATH,
    PROJECT_DIR,
    RUN_DIR,
    _requires_current_us_auth,
    console,
    load_processed_jobs,
    log_run_cost,
    make_evaluator_server,
    make_job_search_server,
    make_scraper_server,
    parse_posting_date,
)
from agentic_job_search.triage import (
    call_mcp_tool,
    chat_openrouter,
    generate_local,
    mcp_session,
    rate_with_ollama,
    rate_with_openrouter,
    triage_job_fit,
    triage_rejects,
)

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    create_sdk_mcp_server,
    query as sdk_query,
    tool,
)


def load_env() -> None:
    env_path = PROJECT_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


load_env()

logger = logging.getLogger(__name__)

BROWSER_PROFILE_DIR = Path.home() / ".linkedin-agent-profile"
TELEGRAM_MCP_URL = 'http://localhost:8004/mcp'
REFERENCE_SUMMARY_CACHE_PATH = RUN_DIR / 'reference_summary_cache.yaml'


async def _send_pipeline_notification(text: str) -> None:
    try:
        await call_mcp_tool(TELEGRAM_MCP_URL, 'send_message', {'text': text})
    except Exception as ex:
        console.print(f'[yellow]Warning: Telegram notification failed: {ex}[/yellow]')


# --- Prompt construction ---

AGENT_INSTRUCTIONS_COMMON = """You are a personal job search agent.

Do not write files directly to disk. Use the save_job_posting tool to persist job data — it is the only way you should store anything.

When browsing LinkedIn:
- Navigate to https://www.linkedin.com/jobs/ to search for jobs
- Extract: job title, company, location, remote/in-person/hybrid (if shown), salary (if shown), and key requirements
- **Before evaluating any job, call `check_and_record_job` with the site, job ID, posting date (YYYY-MM-DD), company, and job title. If it returns `already_processed`, `too_old`, `already_applied`, or `auth_required`, skip the job entirely. Only proceed with jobs that return `new`.**

The user's LinkedIn session is persisted so they should already be logged in. If not, ask them to log in via the browser.

## Hard rule — US jobs without visa sponsorship
If the job is located in the United States, check whether the posting explicitly states that visa sponsorship is available (e.g., "we sponsor visas", "H-1B sponsorship available", "willing to sponsor"). If it does NOT explicitly mention visa sponsorship, rate it **1** immediately.

## Rating jobs

Rate every job you evaluate on a 1–5 scale:
- 1 — Poor fit (missing key requirements or deal-breakers)
- 2 — Weak fit (some relevant aspects but significant gaps)
- 3 — Decent fit (meets most requirements, worth considering)
- 4 — Good fit (strong match on most criteria)
- 5 — Excellent fit (matches nearly everything)
"""

AGENT_INSTRUCTIONS_INTERACTIVE = AGENT_INSTRUCTIONS_COMMON + """
## Interactive mode

Your job is to:
1. Search LinkedIn for relevant job postings based on the user's resume and job requirements
2. Present job listings to the user one at a time and ask if they're interested
3. Learn from feedback (yes/no + reason) to refine future searches and build a picture of the ideal role
4. Eventually help draft cover letters and apply to approved jobs (email, web forms, or LinkedIn)

Present each job clearly, then ask the user if it's a good fit and why.
"""

SCRAPER_INSTRUCTIONS = """You are a job listing scraper. Your job is to find new job postings on LinkedIn and add them to the internal evaluation queue.

You are fully authorized to call all available tools. Call them directly — do not ask for permission.

## CRITICAL RULE: Only use search results pages. NEVER click through to individual job pages.

The information visible in search results (title, company, snippet, date) is all you need. A separate evaluation agent will visit individual job pages later. Your only job is to queue candidates from what you can see in the results list.

## Location targeting

For each search query, run it against both target regions:

1. **Canada (remote)**:
   https://www.linkedin.com/jobs/search/?keywords=<QUERY>&location=Canada&f_WT=2&sortBy=DD

2. **European Union (remote)**:
   https://www.linkedin.com/jobs/search/?keywords=<QUERY>&location=European+Union&f_WT=2&sortBy=DD

(`f_WT=2` = Remote filter. `sortBy=DD` = newest first — critical so fresh postings appear before already-seen ones. URL-encode spaces as `+`. Example for "Principal AI Engineer":
  https://www.linkedin.com/jobs/search/?keywords=Principal+AI+Engineer&location=Canada&f_WT=2&sortBy=DD
  https://www.linkedin.com/jobs/search/?keywords=Principal+AI+Engineer&location=European+Union&f_WT=2&sortBy=DD)

If either location-filtered search returns fewer than 3 new candidates, also run the same query without location filters to catch globally-remote roles that may accept candidates from those regions.

For each job visible in the search results:
1. Call check_and_record_job with site="linkedin", the job ID (from the URL), company, and title. Pass date_posted only if it's visible in the results — it may be relative like "4 days ago", or it may not be shown at all; both are fine.
2. If it returns "new", call queue_candidate immediately with whatever is visible: URL, title, company, snippet, date_posted if shown, and query set to the search query string currently being processed (e.g. "Staff ML Engineer").
3. If it returns "already_processed", "too_old", "already_applied", or "auth_required", skip it.

Only pass information that is directly visible in the search results listing. Do not infer or fabricate missing fields. Stage 2 will navigate to the job page and fill in any missing details.

Run the provided search queries and scan 1–2 pages of results each. Then stop.
Do not evaluate jobs, do not click job titles, do not open job detail pages — just collect candidates from the search results list.
"""

EXTRACTOR_INSTRUCTIONS = """You are a job page extractor. Navigate to the job URL provided and capture a condensed extract of the posting.

Follow EXACTLY this sequence — you have a hard turn budget, and calling submit_job_extract is the only thing that counts as success:
1. browser_navigate to the URL.
2. browser_snapshot.
3. If (and only if) the job description text is visibly cut off and a "… more" / "See more" button exists inside the description: click it, then browser_snapshot once more.
4. Call submit_job_extract IMMEDIATELY with what you have. Do not scroll, do not take additional snapshots, do not search the page, do not run JavaScript, do not verify anything else. A partial description is acceptable — submitting something always beats running out of turns.

If you have a title, company, and any description text, calling submit_job_extract is ALWAYS your next action. Never end a turn without having either taken your one snapshot or submitted the extract.

Condense aggressively: keep the title, company, location, posting date, salary, requirements, responsibilities, tech stack, seniority, and any visa/work-authorization or "no longer accepting applications" statements. In the location field, always include the workplace type shown on the page (Remote / Hybrid / On-site), e.g. "Bucharest, Romania (Remote within country)". Strip navigation chrome, footers, "similar jobs" lists, and marketing boilerplate.

Also capture:
- language_requirement: languages the posting explicitly REQUIRES (not nice-to-haves), comma-separated lowercase, e.g. "english, german". Leave empty if no language requirement is stated.
- relocation: if the posting requires the candidate to relocate to or reside in a specific country/city (e.g. "must be based in Portugal", "remote within Spain", "relocation to Madrid"), give that location. Leave empty for work-from-anywhere roles.

Do not rate the job. Do not browse other pages. Extract this one posting, submit it, then stop.
"""

EVALUATOR_INSTRUCTIONS = """You are evaluating a single job posting. A condensed extract of the posting is provided in the user message — you do not need to browse anywhere.

## Hard rule — US jobs without visa sponsorship
If the job is located in the United States and the posting does NOT explicitly state that visa sponsorship is available (e.g., "we sponsor visas", "H-1B sponsorship available", "willing to sponsor"), rate it **1**.

## Relocation
Relocation to (or residing in) an EU country is acceptable — do NOT reject or heavily penalize a job for requiring it. Treat it as a minor consideration, mention it in your reasoning, and rate primarily on role fit.

Rate the job 1–5 based on the requirements below:
- 1 — Poor fit (missing key requirements or deal-breakers)
- 2 — Weak fit (some relevant aspects but significant gaps)
- 3 — Decent fit (meets most requirements, worth considering)
- 4 — Good fit (strong match on most criteria)
- 5 — Excellent fit (matches nearly everything)

Also produce a 2–3 sentence reasoning and a short label summarising the job (used in the saved filename).
"""

EXTRACT_OUTPUT_SCHEMA = {
    'type': 'object',
    'properties': {
        'title': {'type': 'string'},
        'company': {'type': 'string'},
        'description': {'type': 'string', 'description': 'Condensed posting content (requirements, responsibilities, stack, seniority)'},
        'location': {'type': 'string', 'description': 'Location including workplace type (Remote / Hybrid / On-site)'},
        'date_posted': {'type': 'string', 'description': 'As shown on the page, absolute or relative'},
        'closed': {'type': 'boolean', 'description': 'True if the page shows "No longer accepting applications"'},
        'salary': {'type': 'string'},
        'sponsorship_note': {'type': 'string', 'description': 'Any visa/work-authorization statement, verbatim'},
        'language_requirement': {'type': 'string', 'description': "Explicitly required languages, comma-separated lowercase, e.g. 'english, german'"},
        'relocation': {'type': 'string', 'description': 'Location the candidate must relocate to / reside in, if the posting requires one'},
    },
    'required': ['title', 'company', 'description'],
}

RATING_OUTPUT_SCHEMA = {
    'type': 'object',
    'properties': {
        'rating': {'type': 'integer', 'description': 'Job fit rating, 1-5'},
        'company': {'type': 'string', 'description': 'Hiring company'},
        'title': {'type': 'string', 'description': 'Job title'},
        'reasoning': {'type': 'string', 'description': '2-3 sentences on the fit'},
        'summary': {'type': 'string', 'description': 'Short label for the job, used in the saved filename'},
    },
    'required': ['rating', 'company', 'title', 'reasoning', 'summary'],
}


def load_resume() -> str | None:
    matches = list(RUN_DIR.glob('*-resume-*.md'))
    matches += list(RUN_DIR.glob('*-Resume-*.md'))
    if not matches:
        return None
    latest = max(matches, key=lambda p: p.stat().st_mtime)
    mtime = datetime.fromtimestamp(latest.stat().st_mtime).strftime('%Y-%m-%d')
    console.print(f'[dim]Loaded resume ({len(matches)} found): {latest.name} (modified {mtime})[/dim]')
    return latest.read_text(encoding='utf-8')


def build_system_prompt(interactive: bool) -> str:
    # Static content first (maximises cache-prefix hits across runs).
    parts = []

    if interactive:
        # Resume only needed in interactive mode; non-interactive uses JOB_REQUIREMENTS.md criteria.
        resume = load_resume()
        if resume:
            parts.append(f"--- RESUME ---\n{resume}\n--- END RESUME ---")
        else:
            console.print("[yellow]Warning: no resume file found matching *-resume-*.<md|pdf>[/yellow]")

    if JOB_REQUIREMENTS_PATH.exists():
        requirements = JOB_REQUIREMENTS_PATH.read_text(encoding="utf-8")
        parts.append(f"--- JOB_REQUIREMENTS.md ---\n{requirements}\n--- END JOB_REQUIREMENTS.md ---")

    parts.append(AGENT_INSTRUCTIONS_INTERACTIVE)

    return "\n\n".join(parts)


def build_scraper_prompt() -> str:
    # No date needed — check_and_record_job enforces age filtering via tool.
    parts = []
    if JOB_REQUIREMENTS_PATH.exists():
        requirements = JOB_REQUIREMENTS_PATH.read_text(encoding="utf-8")
        parts.append(f"--- JOB_REQUIREMENTS.md ---\n{requirements}\n--- END JOB_REQUIREMENTS.md ---")
    parts.append(SCRAPER_INSTRUCTIONS)
    return "\n\n".join(parts)


async def generate_search_queries(stage_stats: dict | None = None) -> list[str]:
    """Use Sonnet to derive LinkedIn search queries from resume + job requirements."""
    parts = []
    resume = load_resume()
    if resume:
        parts.append(f'--- RESUME ---\n{resume}\n--- END RESUME ---')
    if JOB_REQUIREMENTS_PATH.exists():
        requirements = JOB_REQUIREMENTS_PATH.read_text(encoding='utf-8')
        parts.append(f'--- JOB_REQUIREMENTS.md ---\n{requirements}\n--- END JOB_REQUIREMENTS.md ---')
    context = '\n\n'.join(parts)

    captured: list[str] = []

    @tool(
        'submit_search_queries',
        'Submit the generated list of job search queries.',
        {
            'type': 'object',
            'properties': {
                'queries': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': '2–6 word LinkedIn search queries, e.g. ["Staff ML Engineer", "Principal AI Engineer"]',
                },
            },
            'required': ['queries'],
        },
    )
    async def _submit(args: dict[str, Any]) -> dict:
        captured.extend(args['queries'])
        return {'content': [{'type': 'text', 'text': 'Queries submitted.'}]}

    query_server = create_sdk_mcp_server(name='query_generator', version='1.0.0', tools=[_submit])

    prompt = (
        f'{context}\n\n'
        'Based on the resume and job requirements above, generate enough short job search queries '
        '(2–6 words each, like a job title) to get good coverage of the most relevant roles — '
        'enough to surface diverse results, but not so many that searches become redundant. '
        'Call the submit_search_queries tool with your list of queries.'
    )
    options = ClaudeAgentOptions(
        tools=[],
        model=MODEL_NAME_LOW,
        system_prompt='You are a tool-calling assistant. Always respond by calling the provided tool — never respond with text.',
        mcp_servers={'query_generator': query_server},
        allowed_tools=['mcp__query_generator__submit_search_queries'],
        permission_mode='bypassPermissions',
        cwd=str(PROJECT_DIR),
    )
    try:
        async for msg in sdk_query(prompt=prompt, options=options):
            if isinstance(msg, ResultMessage):
                print_result_stats(msg)
                if stage_stats is not None:
                    accumulate_stage_stats(stage_stats, msg)
    except Exception as ex:
        raise RuntimeError(f'Stage 1a (query generation) failed: {ex}') from ex

    if not captured:
        raise ValueError('LLM did not call submit_search_queries')
    return captured


def build_profile_block(reference_block: str = '') -> str:
    """Requirements + reference profile, without rating instructions (shared by triage and rating)."""
    parts = []
    if JOB_REQUIREMENTS_PATH.exists():
        requirements = JOB_REQUIREMENTS_PATH.read_text(encoding="utf-8")
        parts.append(f"--- JOB_REQUIREMENTS.md ---\n{requirements}\n--- END JOB_REQUIREMENTS.md ---")
    if reference_block:
        parts.append(reference_block)
    return "\n\n".join(parts)


def build_evaluator_prompt(reference_block: str = '') -> str:
    # Fully static within a run — identical for every job evaluation → maximum prompt caching.
    # No date needed — check_and_record_job enforces age filtering via tool.
    parts = []
    profile = build_profile_block(reference_block)
    if profile:
        parts.append(profile)
    parts.append(EVALUATOR_INSTRUCTIONS)
    # The system prompt is passed to the CLI as a subprocess argument; a stray
    # null byte anywhere in it makes every session fail to spawn.
    return "\n\n".join(parts).replace('\x00', '')


def build_reference_block() -> str:
    texts = tools_module._reference_job_texts
    if not texts:
        return ''
    parts = ['--- REFERENCE JOBS (jobs I have applied to — treat as 5/5 calibration examples) ---']
    for i, text in enumerate(texts[:MAX_REFERENCE_JOBS], 1):
        parts.append(f'[Reference Job {i}]\n{text[:3000]}')
    parts.append('--- END REFERENCE JOBS ---')
    return '\n\n'.join(parts)


async def _summarize_references_anthropic(prompt: str, stage_stats: dict | None) -> str:
    options = ClaudeAgentOptions(
        model=MODEL_NAME_LOW,
        tools=[],
        permission_mode='bypassPermissions',
        output_format={
            'type': 'json_schema',
            'schema': {
                'type': 'object',
                'properties': {'summary': {'type': 'string', 'description': 'The distilled ideal-role profile'}},
                'required': ['summary'],
            },
        },
        cwd=str(PROJECT_DIR),
    )
    async for msg in sdk_query(prompt=prompt, options=options):
        if isinstance(msg, ResultMessage):
            if stage_stats is not None:
                accumulate_stage_stats(stage_stats, msg)
            if msg.structured_output:
                return msg.structured_output['summary']
    raise RuntimeError('Anthropic reference summarization returned no structured output')


async def build_reference_summary(stage_stats: dict | None = None) -> str:
    """Distill the reference jobs into a compact profile block, once per corpus (cached).

    Provider chain: OpenRouter MCP -> local Ollama MCP -> Anthropic Haiku; each failure
    logs and falls through. Falls back to the full reference block if all three fail.
    """
    texts = tools_module._reference_job_texts
    if not texts:
        return ''
    combined = '\n\n'.join(f'[Reference Job {i}]\n{t[:3000]}' for i, t in enumerate(texts[:MAX_REFERENCE_JOBS], 1))
    corpus_hash_md5 = hashlib.md5(combined.encode('utf-8')).hexdigest()

    cache: dict = {}
    if REFERENCE_SUMMARY_CACHE_PATH.exists():
        try:
            cache = yaml.safe_load(REFERENCE_SUMMARY_CACHE_PATH.read_text(encoding='utf-8')) or {}
        except Exception as ex:
            logger.warning(f'Could not read reference summary cache {REFERENCE_SUMMARY_CACHE_PATH}: {ex}')
    if cache.get('hash') == corpus_hash_md5 and cache.get('summary'):
        logger.info(f'Reference summary: cache hit ({len(combined)} chars → {len(cache["summary"])} chars, provider={cache.get("provider", "?")})')
        return _wrap_reference_summary(cache['summary'], len(texts))

    prompt = (
        f'{combined}\n\n'
        f'The postings above are jobs the candidate chose to apply to — treat them as 5/5 fit examples. '
        f'Distill them into ONE "ideal role profile" of at most {REFERENCE_SUMMARY_MAX_CHARS} characters: '
        f'the recurring titles/seniority, domains, tech stack, responsibilities, locations/remote patterns, '
        f'and compensation ranges. Write it as a dense reference profile for calibrating job-fit ratings, '
        f'not as prose about each individual job.'
    )

    summary, provider = '', ''
    try:
        content, cost_usd = await chat_openrouter(prompt)
        summary, provider = content, 'openrouter'
        if stage_stats is not None:
            stage_stats['cost'] += cost_usd
    except Exception as ex:
        logger.warning(f'Reference summary via OpenRouter failed, trying local LLM: {ex}')
        try:
            summary, provider = await generate_local(prompt), 'ollama'
        except Exception as ex2:
            logger.warning(f'Reference summary via local LLM failed, trying Anthropic: {ex2}')
            try:
                summary, provider = await _summarize_references_anthropic(prompt, stage_stats), 'anthropic'
            except Exception as ex3:
                logger.error(f'Reference summary failed on all providers, using full reference block: {ex3}')
                return build_reference_block()

    summary = summary.strip()[:REFERENCE_SUMMARY_MAX_CHARS]
    logger.info(f'Reference summary: {len(combined)} chars → {len(summary)} chars via {provider}')
    REFERENCE_SUMMARY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE_SUMMARY_CACHE_PATH.write_text(
        yaml.dump({'hash': corpus_hash_md5, 'provider': provider, 'summary': summary}, default_flow_style=False),
        encoding='utf-8',
    )
    return _wrap_reference_summary(summary, len(texts))


def _wrap_reference_summary(summary: str, num_jobs: int) -> str:
    return (
        f'--- IDEAL ROLE PROFILE (distilled from {num_jobs} jobs I applied to — treat as 5/5 calibration) ---\n'
        f'{summary}\n'
        f'--- END IDEAL ROLE PROFILE ---'
    )


# --- Runner ---

def print_thinking(text: str) -> None:
    display = text if len(text) <= THINKING_MAX_CHARS else text[:THINKING_MAX_CHARS] + f'\n… ({len(text) - THINKING_MAX_CHARS} more chars)'
    console.print(f"\n[dim italic]Thinking: {display}[/dim italic]\n")


def print_result_stats(msg: ResultMessage) -> None:
    parts = []
    if msg.usage:
        parts.append(f"in={msg.usage.get('input_tokens', 0)} out={msg.usage.get('output_tokens', 0)}")
        cache_read = msg.usage.get("cache_read_input_tokens", 0)
        cache_write = msg.usage.get("cache_creation_input_tokens", 0)
        if cache_read or cache_write:
            parts.append(f"cache_read={cache_read} cache_write={cache_write}")
    if msg.total_cost_usd is not None:
        parts.append(f"cost=${msg.total_cost_usd:.4f}")
    if parts:
        console.print(f"[dim]{' · '.join(parts)}[/dim]")


def new_stage_stats() -> dict:
    return {
        "cost": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }


def accumulate_stage_stats(stats: dict, msg: ResultMessage) -> None:
    if msg.usage:
        stats["input_tokens"] += msg.usage.get("input_tokens", 0)
        stats["output_tokens"] += msg.usage.get("output_tokens", 0)
        stats["cache_read_input_tokens"] += msg.usage.get("cache_read_input_tokens", 0)
        stats["cache_creation_input_tokens"] += msg.usage.get("cache_creation_input_tokens", 0)
    if msg.total_cost_usd is not None:
        stats["cost"] += msg.total_cost_usd


async def run_interactive(client: ClaudeSDKClient) -> None:
    console.print("[bold cyan]Job Search Agent — Interactive Mode[/bold cyan]")
    console.print("[cyan]" + "=" * 40 + "[/cyan]")
    console.print("[dim]Note: On first run, you may need to log in to LinkedIn in the browser window.[/dim]")
    console.print("[dim]Type 'quit' to exit.[/dim]\n")
    console.print("[yellow]Reading your resume and job requirements - please wait ...[/yellow]")

    initial = "You have the resume and JOB_REQUIREMENTS.md in your context. Review them, then ask the user: 'Should I start the search on LinkedIn?'"
    await client.query(initial)

    while True:
        console.print("\n[bold green]Agent:[/bold green] ", end="")
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, ThinkingBlock):
                        print_thinking(block.thinking)
                    elif isinstance(block, TextBlock):
                        print(block.text, flush=True)
            elif isinstance(msg, ResultMessage):
                print_result_stats(msg)

        user_input = input("\n\033[1;34mYou:\033[0m ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        await client.query(user_input)


_RATING_RE = re.compile(r"-rating_(\d+)-")


def count_new_jobs(jobs_before: set[Path]) -> tuple[int, int]:
    """Return (num_evaluated, num_high_rated) for job files created since snapshot."""
    jobs_after = set(RUN_DIR.glob("saved_jobs-*/job_posting-*.md"))
    new_jobs = jobs_after - jobs_before
    num_evaluated = len(new_jobs)
    num_high_rated = sum(
        1 for f in new_jobs
        if (m := _RATING_RE.search(f.name)) and int(m.group(1)) >= 4
    )
    return num_evaluated, num_high_rated


async def run_scraper(client: ClaudeSDKClient, queries: list[str], stage_stats: dict) -> None:
    """Stage 1: haiku scraper collects candidates from LinkedIn search results."""
    query_list = "\n".join(f'- "{q}"' for q in queries)
    try:
        await client.query(
            f"Begin scraping LinkedIn now. Use these search queries:\n{query_list}\n\n"
            "For each query, call check_and_record_job and queue_candidate for each new job found "
            "in the results list. Do not ask for permission — call the tools directly. "
            "Do not navigate to individual job pages. Stop when done."
        )
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, ThinkingBlock):
                        print_thinking(block.thinking)
                    elif isinstance(block, TextBlock):
                        print(block.text, flush=True)
            elif isinstance(msg, ResultMessage):
                print_result_stats(msg)
                accumulate_stage_stats(stage_stats, msg)
    except Exception as ex:
        raise RuntimeError(f'Stage 1b (scraper) failed: {ex}') from ex


async def extract_job_page(candidate: dict, playwright_mcp: dict, stage_stats: dict) -> dict | None:
    """Stage 2a: Haiku agentic session fetches the job page and submits a condensed extract."""
    options = ClaudeAgentOptions(
        tools=[],
        system_prompt=EXTRACTOR_INSTRUCTIONS,
        mcp_servers={
            "playwright": playwright_mcp,
            "job_evaluator": make_evaluator_server(),
        },
        allowed_tools=[
            "mcp__playwright__browser_navigate",
            "mcp__playwright__browser_snapshot",
            "mcp__playwright__browser_click",
            "mcp__job_evaluator__submit_job_extract",
        ],
        permission_mode="bypassPermissions",
        cwd=str(PROJECT_DIR),
        model=MODEL_NAME_LOW,
        max_turns=16,
    )
    query = (
        f"Extract this job posting:\n"
        f"Company: {candidate['company']}\n"
        f"Title: {candidate['title']}\n"
        f"URL: {candidate['url']}\n"
        f"Posted: {candidate['date_posted']}\n"
        f"Snippet: {candidate['snippet']}"
    )
    extracts_before = len(tools_module._job_extracts)
    session_tokens_in = 0
    async with ClaudeSDKClient(options) as client:
        await client.query(query)
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, ThinkingBlock):
                        print_thinking(block.thinking)
                    elif isinstance(block, TextBlock):
                        print(block.text, flush=True)
            elif isinstance(msg, ResultMessage):
                print_result_stats(msg)
                accumulate_stage_stats(stage_stats, msg)
                if msg.usage:
                    session_tokens_in = (
                        msg.usage.get('input_tokens', 0)
                        + msg.usage.get('cache_read_input_tokens', 0)
                        + msg.usage.get('cache_creation_input_tokens', 0)
                    )
    if len(tools_module._job_extracts) <= extracts_before:
        logger.warning(f"Extract: {candidate['company']} — {candidate['title']}: no extract submitted")
        return None
    extract = tools_module._job_extracts[-1]
    logger.info(
        f"Extract: {candidate['company']} — {candidate['title']}: "
        f"~{session_tokens_in} session input tokens → {len(extract['description'])} chars condensed"
    )
    return extract


_MORE_BUTTON_RE = re.compile(r'button "([^"]*\bmore\b[^"]*)" \[ref=(e\d+)\]', re.IGNORECASE)


async def extract_job_page_direct(candidate: dict, playwright_mcp_url: str, stage_stats: dict) -> dict | None:
    """Fallback for a failed agentic extract: fetch the page snapshot directly via the
    Playwright MCP server (full text, no turn budget) and condense it with one
    non-agentic Haiku call. The browser and logged-in profile are identical to the
    agentic path, so bot-detection exposure is unchanged."""
    async with mcp_session(playwright_mcp_url) as call:
        await call('browser_navigate', {'url': candidate['url']})
        await call('browser_wait_for', {'time': 3})  # let the dynamic description render
        snapshot = await call('browser_snapshot', {})
        m = _MORE_BUTTON_RE.search(snapshot)
        if m:
            try:
                await call('browser_click', {'element': m.group(1), 'target': m.group(2)})
                await call('browser_wait_for', {'time': 1})
                snapshot = await call('browser_snapshot', {})
            except Exception as ex:
                logger.warning(f'Extract fallback: expand click failed, using collapsed description: {ex}')

    prompt = (
        'Below is an accessibility snapshot of a job posting page. Condense it into a structured extract. '
        'Keep requirements, responsibilities, tech stack, seniority, and any visa/work-authorization or '
        '"no longer accepting applications" statements. Strip navigation chrome, footers, similar-jobs '
        'lists, and marketing boilerplate.\n\n'
        f'{snapshot[:80000]}'
    )
    options = ClaudeAgentOptions(
        model=MODEL_NAME_LOW,
        tools=[],
        permission_mode='bypassPermissions',
        output_format={'type': 'json_schema', 'schema': EXTRACT_OUTPUT_SCHEMA},
        cwd=str(PROJECT_DIR),
    )
    structured: dict | None = None
    async for msg in sdk_query(prompt=prompt, options=options):
        if isinstance(msg, ResultMessage):
            print_result_stats(msg)
            accumulate_stage_stats(stage_stats, msg)
            if msg.structured_output:
                structured = msg.structured_output
    if structured is None:
        logger.warning(f"Extract fallback: {candidate['company']} — {candidate['title']}: no structured output")
        return None
    extract = {
        'title': structured['title'], 'company': structured['company'],
        'description': structured['description'],
        'location': structured.get('location', ''), 'date_posted': structured.get('date_posted', ''),
        'closed': structured.get('closed', False), 'salary': structured.get('salary', ''),
        'sponsorship_note': structured.get('sponsorship_note', ''),
        'language_requirement': structured.get('language_requirement', ''),
        'relocation': structured.get('relocation', ''),
    }
    logger.info(
        f"Extract fallback: {candidate['company']} — {candidate['title']}: "
        f"{len(snapshot)} snapshot chars → {len(extract['description'])} chars condensed"
    )
    return extract


def format_extract_text(candidate: dict, extract: dict) -> str:
    lines = [
        f"Title: {extract['title']}",
        f"Company: {extract['company']}",
        f"Location: {extract['location']}",
        f"Posted: {extract['date_posted'] or candidate['date_posted']}",
        f"URL: {candidate['url']}",
    ]
    if candidate['snippet']:
        lines.append(f"Search-result snippet: {candidate['snippet']}")
    if extract['salary']:
        lines.append(f"Salary: {extract['salary']}")
    if extract['sponsorship_note']:
        lines.append(f"Sponsorship/authorization note: {extract['sponsorship_note']}")
    if extract.get('language_requirement'):
        lines.append(f"Language requirement: {extract['language_requirement']}")
    if extract.get('relocation'):
        lines.append(f"Relocation required: {extract['relocation']}")
    if extract['closed']:
        lines.append("Status: no longer accepting applications")
    lines.append(f"\n{extract['description']}")
    return '\n'.join(lines)


def apply_hard_rules(candidate: dict, extract: dict) -> str | None:
    """Deterministic auto-reject rules; returns the reason, or None if the job survives."""
    full_text = format_extract_text(candidate, extract)
    if extract['closed'] or 'no longer accepting applications' in full_text.lower():
        return 'posting closed (no longer accepting applications)'
    posted = parse_posting_date(extract['date_posted'] or candidate['date_posted'])
    if posted and (date.today() - posted).days > JOB_STALE_AGE_DAYS:
        return f'posting older than {JOB_STALE_AGE_DAYS} days ({posted.isoformat()})'
    us_located = 'united states' in extract['location'].lower()
    if _requires_current_us_auth(full_text) or (us_located and 'sponsor' not in full_text.lower()):
        return 'US job without explicit visa sponsorship'
    non_english = [
        lang.strip() for lang in re.split(r'[,;/]', extract.get('language_requirement', ''))
        if lang.strip() and 'english' not in lang.lower()
    ]
    if non_english:
        return f'requires non-English language: {", ".join(non_english)}'
    return None


async def _rate_with_anthropic(evaluator_prompt: str, extract_text: str, stage_stats: dict) -> dict:
    options = ClaudeAgentOptions(
        model=MODEL_NAME_MEDIUM,
        effort='low',
        tools=[],
        system_prompt=evaluator_prompt,
        permission_mode='bypassPermissions',
        output_format={'type': 'json_schema', 'schema': RATING_OUTPUT_SCHEMA},
        cwd=str(PROJECT_DIR),
    )
    structured: dict | None = None
    async for msg in sdk_query(prompt=f'Evaluate this job posting:\n\n{extract_text}', options=options):
        if isinstance(msg, ResultMessage):
            print_result_stats(msg)
            accumulate_stage_stats(stage_stats, msg)
            if msg.structured_output:
                structured = msg.structured_output
    if structured is None:
        raise RuntimeError('Anthropic rating call returned no structured output')
    return structured


async def rate_job(evaluator_prompt: str, extract_text: str, stage_stats: dict) -> dict:
    """Stage 2d: one non-agentic rating call, provider selected by RATING_PROVIDER."""
    if RATING_PROVIDER == 'openrouter':
        result, cost_usd = await rate_with_openrouter(evaluator_prompt, f'Evaluate this job posting:\n\n{extract_text}')
        stage_stats['cost'] += cost_usd
        return result
    if RATING_PROVIDER == 'ollama':
        return await rate_with_ollama(evaluator_prompt, f'Evaluate this job posting:\n\n{extract_text}')
    return await _rate_with_anthropic(evaluator_prompt, extract_text, stage_stats)


async def _save_and_notify(
    candidate: dict, rating: int, summary: str, content: str, notify: bool = True, flags: str = ''
) -> None:
    await tools_module.do_save_job_posting(
        company=candidate['company'], description=summary, rating=rating,
        content=content, job_id=candidate['job_id'],
    )
    if notify and rating >= 4:
        flags_line = f"\n⚠ {flags}" if flags else ''
        await _send_pipeline_notification(
            f"⭐ Job match ({rating}/5): {candidate['company']} — {candidate['title']}\n{summary}{flags_line}\n{candidate['url']}"
        )


async def evaluate_all_candidates(
    candidates: list[dict], playwright_mcp: dict, evaluator_prompt: str,
    profile_block: str, stage_stats: dict,
) -> None:
    """Stage 2: per candidate — Haiku extract, deterministic hard rules, local triage,
    then one configurable-model rating call. All sharing one browser."""
    for candidate in candidates:
        console.print(f"[dim]Evaluating: {candidate['company']} — {candidate['title']}[/dim]")
        try:
            if EXTRACTOR_PROVIDER == 'openrouter':
                extract = await extract_job_page_openrouter(
                    candidate, playwright_mcp['url'], stage_stats['extraction'], EXTRACTOR_INSTRUCTIONS
                )
            else:
                extract = await extract_job_page(candidate, playwright_mcp, stage_stats['extraction'])
            if extract is None:
                extract = await extract_job_page_direct(candidate, playwright_mcp['url'], stage_stats['extraction'])
            if extract is None:
                continue
            extract_text = format_extract_text(candidate, extract)

            hard_rule_reason = apply_hard_rules(candidate, extract)
            if hard_rule_reason:
                logger.info(f"Hard rule: {candidate['company']} — {candidate['title']}: rated 1 ({hard_rule_reason})")
                await _save_and_notify(
                    candidate, rating=1, summary=f"auto rejected {hard_rule_reason}",
                    content=f"# Auto-rated 1 — {hard_rule_reason}\n\n{extract_text}", notify=False,
                )
                continue

            triage_result = None
            if TRIAGE_ENABLED:
                triage_result = await triage_job_fit(extract_text, profile_block)
                if triage_result:
                    logger.info(
                        f"Triage: {candidate['company']} — {candidate['title']}: "
                        f"score {triage_result['score']} ({triage_result['reason']})"
                    )
                if triage_rejects(triage_result):
                    await _save_and_notify(
                        candidate, rating=triage_result['score'],
                        summary=f"triaged out {triage_result['reason']}",
                        content=(
                            f"# Triaged out (local score {triage_result['score']}/5)\n\n"
                            f"Reason: {triage_result['reason']}\n\n{extract_text}"
                        ),
                        notify=False,
                    )
                    continue

            result = await rate_job(evaluator_prompt, extract_text, stage_stats['rating'])
            triage_note = f" (triage said {triage_result['score']})" if triage_result else ''
            logger.info(
                f"Rating: {candidate['company']} — {candidate['title']}: "
                f"{result['rating']}/5 via {RATING_PROVIDER}{triage_note} — {result['reasoning']}"
            )
            relocation_flag = f"relocation required: {extract['relocation']}" if extract.get('relocation') else ''
            content = (
                f"# {result['title']} at {result['company']} — rating {result['rating']}/5\n\n"
                + (f"⚠ {relocation_flag}\n" if relocation_flag else '')
                + f"Reasoning: {result['reasoning']}\n"
                + (f"Triage score (local): {triage_result['score']}\n" if triage_result else '')
                + f"\n{extract_text}"
            )
            await _save_and_notify(
                candidate, rating=result['rating'], summary=result['summary'],
                content=content, flags=relocation_flag,
            )
        except Exception as ex:
            console.print(f"[red]Stage 2 error evaluating {candidate['company']} — {candidate['title']}: {ex}[/red]")


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


async def start_playwright_server(port: int, browser_mode: str = 'minimized') -> asyncio.subprocess.Process:
    cmd = [
        'npx', '@playwright/mcp@latest',
        '--port', str(port),
        '--user-data-dir', str(BROWSER_PROFILE_DIR),
        '--shared-browser-context',
    ]
    tmp_config: str | None = None
    if browser_mode == 'headless':
        cmd.append('--headless')
    elif browser_mode == 'minimized':
        # Pass --start-minimized via a temp config file (no direct CLI flag for launch args)
        config = {'browser': {'launchOptions': {'args': ['--start-minimized']}}}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            tmp_config = f.name
        cmd += ['--config', tmp_config]
    # 'visible': no additional flags
    # @playwright/mcp prints a "Listening on ..." banner + client-config JSON to stdout
    # whenever --port is used; there's no CLI flag or env var to suppress it (checked
    # `npx @playwright/mcp@latest --help`). Harmless noise — the URL below is what we
    # actually use, not the banner's copy-paste config.
    proc = await asyncio.create_subprocess_exec(*cmd)
    try:
        console.print(f'[dim]→ GET http://localhost:{port}/mcp (polling until ready)[/dim]')
        for _ in range(30):
            await asyncio.sleep(1)
            try:
                requests.get(f'http://localhost:{port}/mcp', timeout=1)
                break
            except Exception:
                pass
    finally:
        if tmp_config:
            Path(tmp_config).unlink(missing_ok=True)
    return proc


async def run_non_interactive(browser_mode: str = 'headless') -> None:
    tools_module._candidates = []
    tools_module._candidates_per_query = {}
    tools_module._job_extracts = []

    console.print("[bold cyan]Job Search Agent — Non-interactive Mode[/bold cyan]")
    console.print("[cyan]" + "=" * 40 + "[/cyan]")

    start_time = time.time()
    stage_stats = {
        "query_generation": new_stage_stats(),
        "scraping": new_stage_stats(),
        "reference_summary": new_stage_stats(),
        "extraction": new_stage_stats(),
        "rating": new_stage_stats(),
    }
    jobs_before = set(RUN_DIR.glob("saved_jobs-*/job_posting-*.md"))

    port = find_free_port()
    console.print(f"[dim]Starting shared browser ({browser_mode}, port {port}) ...[/dim]")
    playwright_proc = await start_playwright_server(port, browser_mode=browser_mode)
    playwright_mcp = {'type': 'http', 'url': f'http://localhost:{port}/mcp'}

    try:
        # Stage 1: sonnet generates search queries, haiku scraper does the actual scraping
        console.print("[yellow]Stage 1a: Generating search queries ...[/yellow]")
        queries = await generate_search_queries(stage_stats["query_generation"])
        if not queries:
            console.print("[red]Error: no search queries generated.[/red]")
            await _send_pipeline_notification("Job search run FAILED\n• Error: no search queries generated")
            log_run_cost({
                "timestamp": datetime.now().isoformat(),
                "mode": "non-interactive",
                "status": "failed_no_queries",
                "stage_stats": stage_stats,
                "total_cost": sum(s["cost"] for s in stage_stats.values()),
                "elapsed_minutes": (time.time() - start_time) / 60,
            })
            return
        console.print(f"[dim]Queries: {queries}[/dim]\n")

        console.print("[yellow]Stage 1b: Scraping LinkedIn for candidates ...[/yellow]\n")
        scraper_options = ClaudeAgentOptions(
            tools=[],
            system_prompt=build_scraper_prompt(),
            mcp_servers={
                "playwright": playwright_mcp,
                "job_scraper": make_scraper_server(),
            },
            permission_mode="bypassPermissions",
            cwd=str(PROJECT_DIR),
            model=MODEL_NAME_LOW,
            max_turns=40,
        )
        async with ClaudeSDKClient(scraper_options) as scraper:
            await run_scraper(scraper, queries, stage_stats["scraping"])

        candidates = tools_module._candidates
        candidates_per_query = tools_module._candidates_per_query
        query_summary_lines = [
            f'  "{q}": {candidates_per_query.get(q, 0)} job(s)' + (' ⚠' if candidates_per_query.get(q, 0) == 0 else '')
            for q in queries
        ]
        console.print(f"\n[dim]Stage 1 complete: {len(candidates)} candidate(s) queued.[/dim]")
        for line in query_summary_lines:
            console.print(f"[dim]{line}[/dim]")
        console.print()

        if not candidates:
            console.print("[red]Error: 0 jobs returned across all searches.[/red]")
            elapsed_mins = (time.time() - start_time) / 60
            total_cost = sum(s["cost"] for s in stage_stats.values())
            query_lines = "\n".join(f'  - "{q}": {candidates_per_query.get(q, 0)} jobs' for q in queries)
            await _send_pipeline_notification(
                f"Job search run FAILED\n• Error: 0 candidates found\n• Queries ({len(queries)}):\n{query_lines}\n• Elapsed: {elapsed_mins:.1f} min\n• Total cost: ${total_cost:.4f}"
            )
            log_run_cost({
                "timestamp": datetime.now().isoformat(),
                "mode": "non-interactive",
                "status": "failed_no_candidates",
                "stage_stats": stage_stats,
                "total_cost": total_cost,
                "elapsed_minutes": elapsed_mins,
            })
            return

        # Stage 2: all evaluations share the same browser via the SSE server
        console.print("[yellow]Stage 2: Evaluating candidates ...[/yellow]\n")
        reference_block = await build_reference_summary(stage_stats["reference_summary"])
        evaluator_prompt = build_evaluator_prompt(reference_block)
        profile_block = build_profile_block(reference_block)
        await evaluate_all_candidates(candidates, playwright_mcp, evaluator_prompt, profile_block, stage_stats)
    finally:
        playwright_proc.terminate()
        await playwright_proc.wait()

    elapsed_mins = (time.time() - start_time) / 60
    num_evaluated, num_high_rated = count_new_jobs(jobs_before)
    total_cost = sum(s["cost"] for s in stage_stats.values())

    query_lines = "\n".join(
        f'  - "{q}": {candidates_per_query.get(q, 0)} jobs' + (' ⚠' if candidates_per_query.get(q, 0) == 0 else '')
        for q in queries
    )
    stage_cost_lines = "\n".join(
        f'  - {name}: ${stats["cost"]:.4f}' for name, stats in stage_stats.items()
    )
    stats_lines = [
        "Job search run complete",
        f"• Queries ({len(queries)}):\n{query_lines}",
        f"• Candidates found: {len(candidates)}",
        f"• Jobs saved: {num_evaluated}",
        f"• Jobs rated ≥4: {num_high_rated}",
        f"• Elapsed: {elapsed_mins:.1f} min",
        f"• Cost by stage:\n{stage_cost_lines}",
        f"• Total cost: ${total_cost:.4f}",
    ]
    stats_msg = "\n".join(stats_lines)
    console.print(f"\n[dim]{stats_msg}[/dim]")
    await _send_pipeline_notification(stats_msg)
    log_run_cost({
        "timestamp": datetime.now().isoformat(),
        "mode": "non-interactive",
        "status": "complete",
        "stage_stats": stage_stats,
        "total_cost": total_cost,
        "candidates_found": len(candidates),
        "jobs_saved": num_evaluated,
        "jobs_rated_high": num_high_rated,
        "elapsed_minutes": elapsed_mins,
    })


async def main() -> None:
    # Console + persistent per-run file log (triage/hard-rule rejections, extract sizes,
    # ratings — everything logged via the logging module lands in both).
    log_dir = RUN_DIR / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = log_dir / f'run-{datetime.now().strftime("%Y%b%d-%H%M%S")}.log'
    logging.basicConfig(
        level=os.environ.get('LOG_LEVEL', 'INFO'),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(run_log_path)],
    )
    logging.getLogger('claude_agent_sdk').setLevel(logging.WARNING)
    logger.info(f'Logging to {run_log_path}')
    parser = argparse.ArgumentParser(description="Job Search Agent")
    parser.add_argument(
        "--non-interactive", "-n",
        action="store_true",
        help="Run autonomously: search LinkedIn, rate jobs, notify on 4+, no user interaction",
    )
    parser.add_argument(
        "--browser",
        choices=["headless", "minimized", "visible"],
        default="headless",
        help="Browser display mode for non-interactive runs (default: headless)",
    )
    args = parser.parse_args()
    interactive = not args.non_interactive

    if not interactive and not JOB_REQUIREMENTS_PATH.exists():
        console.print("[red]Error: JOB_REQUIREMENTS.md not found. Non-interactive mode requires it.[/red]")
        sys.exit(1)

    load_processed_jobs()
    await tools_module.categorize_downloads_pdfs()
    await tools_module.load_downloads_applied_pdfs()

    if interactive:
        options = ClaudeAgentOptions(
            system_prompt=build_system_prompt(interactive=True),
            mcp_servers={
                "playwright": {
                    "type": "stdio",
                    "command": "npx",
                    "args": ["@playwright/mcp@latest", "--user-data-dir", str(BROWSER_PROFILE_DIR)],
                },
                "job_search": make_job_search_server(interactive=True),
            },
            permission_mode="acceptEdits",
            cwd=str(PROJECT_DIR),
        )
        async with ClaudeSDKClient(options) as client:
            await run_interactive(client)
    else:
        await run_non_interactive(browser_mode=args.browser)


def cli() -> None:
    asyncio.run(main())
