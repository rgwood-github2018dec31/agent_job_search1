import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import pypdf

from agentic_job_search.config import MODEL_NAME_LOW, MODEL_NAME_MEDIUM
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
3. If it returns "already_processed" or "too_old", skip it.

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
    matches = list(RUN_DIR.glob("*-resume-*.md")) + list(RUN_DIR.glob("*-resume-*.pdf"))
    if not matches:
        return None
    md_matches = [p for p in matches if p.suffix.lower() == ".md"]
    if md_matches:
        latest = max(md_matches, key=lambda p: p.stat().st_mtime)
    else:
        latest = max(matches, key=lambda p: p.stat().st_mtime)
    console.print(f"[dim]Loaded resume: {latest.name}[/dim]")
    if latest.suffix.lower() == ".pdf":
        reader = pypdf.PdfReader(latest)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return latest.read_text(encoding="utf-8")


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
        parts.append(f"--- RESUME ---\n{resume}\n--- END RESUME ---")
    if JOB_REQUIREMENTS_PATH.exists():
        requirements = JOB_REQUIREMENTS_PATH.read_text(encoding="utf-8")
        parts.append(f"--- JOB_REQUIREMENTS.md ---\n{requirements}\n--- END JOB_REQUIREMENTS.md ---")
    context = "\n\n".join(parts)
    prompt = (
        f"{context}\n\n"
        "Based on the resume and job requirements above, generate enough short job search queries "
        "(2–6 words each, like a job title) to get good coverage of the most relevant roles — "
        "enough to surface diverse results, but not so many that searches become redundant. "
        'Reply with ONLY a JSON array of strings, e.g. ["Principal AI Engineer", "Staff ML Engineer"].'
    )
    options = ClaudeAgentOptions(
        model=MODEL_NAME_MEDIUM,
        permission_mode="acceptEdits",
        cwd=str(PROJECT_DIR),
    )
    text_parts: list[str] = []
    async with ClaudeSDKClient(options) as client:
        await client.query(prompt)
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)
    raw = "".join(text_parts)
    console.print(f"[dim]Query generation response: {raw[:200]}[/dim]")
    # Extract JSON array even if wrapped in a markdown code fence
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON array found in query generation response: {raw!r}")
    return json.loads(m.group())


def build_evaluator_prompt() -> str:
    # Fully static — identical for every job evaluation → maximum prompt caching.
    # No date needed — check_and_record_job enforces age filtering via tool.
    parts = []
    if JOB_REQUIREMENTS_PATH.exists():
        requirements = JOB_REQUIREMENTS_PATH.read_text(encoding="utf-8")
        parts.append(f"--- JOB_REQUIREMENTS.md ---\n{requirements}\n--- END JOB_REQUIREMENTS.md ---")
    parts.append(EVALUATOR_INSTRUCTIONS)
    return "\n\n".join(parts)


# --- Runner ---

def print_thinking(text: str) -> None:
    MAX_CHARS = 1000
    display = text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + f"\n… ({len(text) - MAX_CHARS} more chars)"
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
    return cost


async def evaluate_candidate(candidate: dict, playwright_mcp: dict, evaluator_prompt: str) -> float:
    """Stage 2: fresh sonnet session evaluates one job posting."""
    console.print(f"[dim]Evaluating: {candidate['company']} — {candidate['title']}[/dim]")
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
    async with ClaudeSDKClient(options) as client:
        await client.query(
            f"Evaluate this job posting:\n"
            f"Company: {candidate['company']}\n"
            f"Title: {candidate['title']}\n"
            f"URL: {candidate['url']}\n"
            f"Posted: {candidate['date_posted']}\n"
            f"Snippet: {candidate['snippet']}\n\n"
            f"Navigate to the URL, read the full description, rate it 1–5, save it with save_job_posting, "
            f"and call notify_user if rated 4 or 5."
        )
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
    return cost


async def run_non_interactive() -> None:
    tools_module._candidates = []

    console.print("[bold cyan]Job Search Agent — Non-interactive Mode[/bold cyan]")
    console.print("[cyan]" + "=" * 40 + "[/cyan]")

    start_time = time.time()
    total_cost = 0.0
    jobs_before = set(RUN_DIR.glob("saved_jobs-*/job_posting-*.md"))

    playwright_mcp = {
        "type": "stdio",
        "command": "npx",
        "args": ["@playwright/mcp@latest", "--user-data-dir", str(BROWSER_PROFILE_DIR)],
    }

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

    # Stage 2: sonnet evaluator — one fresh session per job, system prompt cached after job 1
    console.print("[yellow]Stage 2: Evaluating candidates ...[/yellow]\n")
    evaluator_prompt = build_evaluator_prompt()  # built once, reused for all jobs
    for candidate in candidates:
        total_cost += await evaluate_candidate(candidate, playwright_mcp, evaluator_prompt)

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
    args = parser.parse_args()
    interactive = not args.non_interactive

    if not interactive and not JOB_REQUIREMENTS_PATH.exists():
        console.print("[red]Error: JOB_REQUIREMENTS.md not found. Non-interactive mode requires it.[/red]")
        sys.exit(1)

    load_processed_jobs()

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
        await run_non_interactive()


def cli() -> None:
    asyncio.run(main())
