import argparse
import asyncio
import json
import os
import re
import socket
import sys
import tempfile
import time
import requests
from pathlib import Path
from typing import Any

from agentic_job_search.config import MAX_REFERENCE_JOBS, MODEL_NAME_LOW, MODEL_NAME_MEDIUM, THINKING_MAX_CHARS
import agentic_job_search.tools_generic as tools_module
from agentic_job_search.tools_generic import (
    JOB_REQUIREMENTS_PATH,
    PROJECT_DIR,
    RUN_DIR,
    console,
    load_processed_jobs,
    make_evaluator_server,
    make_job_search_server,
    make_scraper_server,
    send_telegram,
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

BROWSER_PROFILE_DIR = Path.home() / ".linkedin-agent-profile"


# --- Prompt construction ---

AGENT_INSTRUCTIONS_COMMON = """You are a personal job search agent.

Do not write files directly to disk. Use the save_job_posting tool to persist job data — it is the only way you should store anything.

When browsing LinkedIn:
- Navigate to https://www.linkedin.com/jobs/ to search for jobs
- Extract: job title, company, location, remote/in-person/hybrid (if shown), salary (if shown), and key requirements
- **Before evaluating any job, call `check_and_record_job` with the site, job ID, posting date (YYYY-MM-DD), company, and job title. If it returns `already_processed` or `too_old`, skip the job entirely. Only proceed with jobs that return `new`.**

The user's LinkedIn session is persisted so they should already be logged in. If not, ask them to log in via the browser.

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

For each job visible in the search results:
1. Call check_and_record_job with site="linkedin", the job ID (from the URL), company, and title. Pass date_posted only if it's visible in the results — it may be relative like "4 days ago", or it may not be shown at all; both are fine.
2. If it returns "new", call queue_candidate immediately with whatever is visible: URL, title, company, snippet, and date_posted if shown.
3. If it returns "already_processed", "too_old", or "already_applied", skip it.

Only pass information that is directly visible in the search results listing. Do not infer or fabricate missing fields. Stage 2 will navigate to the job page and fill in any missing details.

Run the provided search queries and scan 1–2 pages of results each. Then stop.
Do not evaluate jobs, do not click job titles, do not open job detail pages — just collect candidates from the search results list.
"""

EVALUATOR_INSTRUCTIONS = """You are evaluating a single job posting.

Navigate to the job URL provided. Read the full job description carefully.

Rate the job 1–5 based on the requirements below:
- 1 — Poor fit (missing key requirements or deal-breakers)
- 2 — Weak fit (some relevant aspects but significant gaps)
- 3 — Decent fit (meets most requirements, worth considering)
- 4 — Good fit (strong match on most criteria)
- 5 — Excellent fit (matches nearly everything)

Then:
1. Call save_job_posting with the full job content, your rating, company, title, and job_id.
2. If rating is 4 or 5, call notify_user with a brief summary.

Evaluate only this one job, then stop. Do not browse other pages.
"""


def load_resume() -> str | None:
    matches = list(RUN_DIR.glob('*-resume-*.md'))
    matches += list(RUN_DIR.glob('*-Resume-*.md'))
    if not matches:
        return None
    latest = max(matches, key=lambda p: p.stat().st_mtime)
    console.print(f'[dim]Loaded resume: {latest.name}[/dim]')
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


async def generate_search_queries() -> list[str]:
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
        model=MODEL_NAME_MEDIUM,
        system_prompt='You are a tool-calling assistant. Always respond by calling the provided tool — never respond with text.',
        mcp_servers={'query_generator': query_server},
        allowed_tools=['mcp__query_generator__submit_search_queries'],
        permission_mode='bypassPermissions',
        cwd=str(PROJECT_DIR),
    )
    try:
        async for _ in sdk_query(prompt=prompt, options=options):
            pass
    except Exception as ex:
        raise RuntimeError(f'Stage 1a (query generation) failed: {ex}') from ex

    if not captured:
        raise ValueError('LLM did not call submit_search_queries')
    return captured


def build_evaluator_prompt() -> str:
    # Fully static — identical for every job evaluation → maximum prompt caching.
    # No date needed — check_and_record_job enforces age filtering via tool.
    parts = []
    if JOB_REQUIREMENTS_PATH.exists():
        requirements = JOB_REQUIREMENTS_PATH.read_text(encoding="utf-8")
        parts.append(f"--- JOB_REQUIREMENTS.md ---\n{requirements}\n--- END JOB_REQUIREMENTS.md ---")
    parts.append(EVALUATOR_INSTRUCTIONS)
    return "\n\n".join(parts)


def build_reference_block() -> str:
    texts = tools_module._reference_job_texts
    if not texts:
        return ''
    parts = ['--- REFERENCE JOBS (jobs I have applied to — treat as 5/5 calibration examples) ---']
    for i, text in enumerate(texts[:MAX_REFERENCE_JOBS], 1):
        parts.append(f'[Reference Job {i}]\n{text[:3000]}')
    parts.append('--- END REFERENCE JOBS ---')
    return '\n\n'.join(parts)


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
                        print(block.text, end="", flush=True)
            elif isinstance(msg, ResultMessage):
                print()
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


async def run_scraper(client: ClaudeSDKClient, queries: list[str]) -> float:
    """Stage 1: haiku scraper collects candidates from LinkedIn search results."""
    query_list = "\n".join(f'- "{q}"' for q in queries)
    try:
        await client.query(
            f"Begin scraping LinkedIn now. Use these search queries:\n{query_list}\n\n"
            "For each query, call check_and_record_job and queue_candidate for each new job found "
            "in the results list. Do not ask for permission — call the tools directly. "
            "Do not navigate to individual job pages. Stop when done."
        )
        cost = 0.0
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, ThinkingBlock):
                        print_thinking(block.thinking)
                    elif isinstance(block, TextBlock):
                        print(block.text, end="", flush=True)
            elif isinstance(msg, ResultMessage):
                print()
                print_result_stats(msg)
                if msg.total_cost_usd is not None:
                    cost += msg.total_cost_usd
    except Exception as ex:
        raise RuntimeError(f'Stage 1b (scraper) failed: {ex}') from ex
    return cost


async def evaluate_all_candidates(
    candidates: list[dict], playwright_mcp: dict, evaluator_prompt: str, reference_block: str = ''
) -> float:
    """Stage 2: single session evaluates all job postings, reusing one browser."""
    options = ClaudeAgentOptions(
        system_prompt=evaluator_prompt,
        mcp_servers={
            "playwright": playwright_mcp,
            "job_evaluator": make_evaluator_server(),
        },
        permission_mode="bypassPermissions",
        cwd=str(PROJECT_DIR),
        effort="low",
    )
    cost = 0.0
    for candidate in candidates:
        console.print(f"[dim]Evaluating: {candidate['company']} — {candidate['title']}[/dim]")
        query = (
            f"Evaluate this job posting:\n"
            f"Company: {candidate['company']}\n"
            f"Title: {candidate['title']}\n"
            f"URL: {candidate['url']}\n"
            f"Posted: {candidate['date_posted']}\n"
            f"Snippet: {candidate['snippet']}\n\n"
            f"Navigate to the URL, read the full description, rate it 1–5, save it with save_job_posting, "
            f"and call notify_user if rated 4 or 5."
        )
        if reference_block:
            query = f'{reference_block}\n\n{query}'
        try:
            async with ClaudeSDKClient(options) as client:
                await client.query(query)
                async for msg in client.receive_response():
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, ThinkingBlock):
                                print_thinking(block.thinking)
                            elif isinstance(block, TextBlock):
                                print(block.text, end="", flush=True)
                    elif isinstance(msg, ResultMessage):
                        print()
                        print_result_stats(msg)
                        if msg.total_cost_usd is not None:
                            cost += msg.total_cost_usd
        except Exception as ex:
            console.print(f"[red]Stage 2 error evaluating {candidate['company']} — {candidate['title']}: {ex}[/red]")
    return cost


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

    console.print("[bold cyan]Job Search Agent — Non-interactive Mode[/bold cyan]")
    console.print("[cyan]" + "=" * 40 + "[/cyan]")

    start_time = time.time()
    total_cost = 0.0
    jobs_before = set(RUN_DIR.glob("saved_jobs-*/job_posting-*.md"))

    port = find_free_port()
    console.print(f"[dim]Starting shared browser ({browser_mode}, port {port}) ...[/dim]")
    playwright_proc = await start_playwright_server(port, browser_mode=browser_mode)
    playwright_mcp = {'type': 'http', 'url': f'http://localhost:{port}/mcp'}

    try:
        # Stage 1: sonnet generates search queries, haiku scraper does the actual scraping
        console.print("[yellow]Stage 1a: Generating search queries ...[/yellow]")
        queries = await generate_search_queries()
        console.print(f"[dim]Queries: {queries}[/dim]\n")

        console.print("[yellow]Stage 1b: Scraping LinkedIn for candidates ...[/yellow]\n")
        scraper_options = ClaudeAgentOptions(
            system_prompt=build_scraper_prompt(),
            mcp_servers={
                "playwright": playwright_mcp,
                "job_scraper": make_scraper_server(),
            },
            permission_mode="bypassPermissions",
            cwd=str(PROJECT_DIR),
            model=MODEL_NAME_LOW,
        )
        async with ClaudeSDKClient(scraper_options) as scraper:
            total_cost += await run_scraper(scraper, queries)

        candidates = tools_module._candidates
        console.print(f"\n[dim]Stage 1 complete: {len(candidates)} candidate(s) queued.[/dim]\n")

        if not candidates:
            console.print("[dim]No new candidates found.[/dim]")
            elapsed_mins = (time.time() - start_time) / 60
            send_telegram(f"Job search run complete\n• No new candidates\n• Elapsed: {elapsed_mins:.1f} min\n• Total cost: ${total_cost:.4f}")
            return

        # Stage 2: all evaluations share the same browser via the SSE server
        console.print("[yellow]Stage 2: Evaluating candidates ...[/yellow]\n")
        evaluator_prompt = build_evaluator_prompt()
        reference_block = build_reference_block()
        total_cost += await evaluate_all_candidates(candidates, playwright_mcp, evaluator_prompt, reference_block)
    finally:
        playwright_proc.terminate()
        await playwright_proc.wait()

    elapsed_mins = (time.time() - start_time) / 60
    num_evaluated, num_high_rated = count_new_jobs(jobs_before)

    stats_lines = [
        "Job search run complete",
        f"• Candidates found: {len(candidates)}",
        f"• Jobs saved: {num_evaluated}",
        f"• Jobs rated ≥4: {num_high_rated}",
        f"• Elapsed: {elapsed_mins:.1f} min",
        f"• Total cost: ${total_cost:.4f}",
    ]
    stats_msg = "\n".join(stats_lines)
    console.print(f"\n[dim]{stats_msg}[/dim]")
    try:
        send_telegram(stats_msg)
    except Exception as e:
        console.print(f"[yellow]Warning: failed to send stats via Telegram: {e}[/yellow]")


async def main() -> None:
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
