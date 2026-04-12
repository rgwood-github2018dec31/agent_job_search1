import argparse
import asyncio
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pypdf
import yaml
from rich.console import Console

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    create_sdk_mcp_server,
    tool,
)


def load_env() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


load_env()

console = Console()

PROJECT_DIR = Path(__file__).parent
JOB_REQUIREMENTS_PATH = PROJECT_DIR / "JOB_REQUIREMENTS.md"
PROCESSED_JOBS_DIR = PROJECT_DIR / "processed_jobs"
BROWSER_PROFILE_DIR = Path.home() / ".linkedin-agent-profile"

_processed_jobs: set[tuple[str, str]] = set()
_candidates: list[dict] = []


def underscorify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def send_telegram(text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        console.print("[yellow]Warning: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set, skipping notification.[/yellow]")
        return
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data=data)


# Old MD filename pattern: job_posting-{linkedin_id}[-rating_{N}]-{company}-{desc}-{timestamp}.md
_SAVED_JOB_MD_RE = re.compile(r"^job_posting-(\d+|noid)(?:-rating_\d+)?-.+-\d+\.md$")


def load_processed_jobs() -> None:
    """Populate _processed_jobs from both old MD filenames and new per-job YAML files."""
    global _processed_jobs
    ids: set[tuple[str, str]] = set()

    # Old format: extract (linkedin, job_id) from saved MD filenames
    for f in PROJECT_DIR.glob("saved_jobs-*/job_posting-*.md"):
        m = _SAVED_JOB_MD_RE.match(f.name)
        if m:
            job_id = m.group(1)
            if job_id != "noid":
                ids.add(("linkedin", job_id))

    # New format: load (site, job_id) from per-job YAML files
    for f in PROCESSED_JOBS_DIR.glob("*.yaml"):
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
            if data and "site" in data and "job_id" in data:
                ids.add((data["site"], str(data["job_id"])))
        except Exception:
            pass

    _processed_jobs = ids
    console.print(f"[dim]Loaded {len(_processed_jobs)} previously processed job(s).[/dim]")


# --- Tool implementations (plain async functions, directly testable) ---

async def do_save_job_posting(
    company: str, description: str, rating: int, content: str, job_id: str | None = None
) -> dict:
    date_str = datetime.now().strftime("%Y%b%d")
    dir_path = PROJECT_DIR / f"saved_jobs-{date_str}"
    dir_path.mkdir(exist_ok=True)

    ts = int(time.time())
    id_part = job_id if job_id else "noid"
    rating_part = f"-rating_{rating}" if rating is not None else ""
    filename = f"job_posting-{id_part}{rating_part}-{underscorify(company)}-{underscorify(description)}-{ts}.md"
    (dir_path / filename).write_text(content, encoding="utf-8")

    return {"content": [{"type": "text", "text": f"Saved: saved_jobs-{date_str}/{filename}"}]}


async def do_notify_user(message: str) -> dict:
    try:
        send_telegram(message)
        return {"content": [{"type": "text", "text": "Notification sent."}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Notification failed: {e}"}], "isError": True}


async def do_update_job_requirements(content: str) -> dict:
    JOB_REQUIREMENTS_PATH.write_text(content, encoding="utf-8")
    return {
        "content": [
            {"type": "text", "text": f"JOB_REQUIREMENTS.md updated. New contents:\n\n{content}"}
        ]
    }


def parse_posting_date(date_posted: str | None) -> date | None:
    """Parse an absolute (YYYY-MM-DD) or relative ('4 days ago') posting date."""
    if not date_posted:
        return None
    s = date_posted.strip().lower()
    # Absolute ISO date
    try:
        return date.fromisoformat(s)
    except ValueError:
        pass
    # Relative: "N unit(s) ago"
    m = re.match(r"(\d+)\s+(hour|day|week|month)s?\s+ago", s)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        deltas = {"hour": timedelta(hours=n), "day": timedelta(days=n),
                  "week": timedelta(weeks=n), "month": timedelta(days=n * 30)}
        return (datetime.now() - deltas[unit]).date()
    # "just now" / "today"
    if s in ("just now", "today", "moments ago"):
        return date.today()
    return None


async def do_check_and_record_job(
    site: str, job_id: str, company: str, description: str,
    date_posted: str | None = None,
    url: str | None = None, content: str | None = None
) -> dict:
    key = (site, job_id)
    if key in _processed_jobs:
        return {"content": [{"type": "text", "text": "already_processed"}]}

    posted = parse_posting_date(date_posted)
    if posted and (date.today() - posted).days > 21:
        return {"content": [{"type": "text", "text": "too_old"}]}

    PROCESSED_JOBS_DIR.mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%Y%b%d")
    ts = int(time.time())
    filename = f"job_posting-{site}-{job_id}-{date_str}-{ts}-{underscorify(company)}-{underscorify(description)}.yaml"
    data = {
        "site": site,
        "job_id": job_id,
        "date_posted": date_posted,
        "date_recorded": date.today().isoformat(),
        "company": company,
        "description": description,
    }
    if url:
        data["url"] = url
    if content:
        data["content"] = content
    (PROCESSED_JOBS_DIR / filename).write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
    _processed_jobs.add(key)
    return {"content": [{"type": "text", "text": "new"}]}


async def do_queue_candidate(
    site: str, job_id: str, url: str, title: str, company: str,
    snippet: str, date_posted: str | None = None,
) -> dict:
    _candidates.append({
        "site": site, "job_id": job_id, "url": url,
        "title": title, "company": company,
        "date_posted": date_posted or "", "snippet": snippet,
    })
    return {"content": [{"type": "text", "text": f"Queued: {company} — {title}"}]}


# --- Tool wrappers (SDK @tool decorators delegate to the implementations above) ---

@tool(
    "save_job_posting",
    "Save a job posting to the daily saved_jobs directory. Call this for every job evaluated. "
    "Extract the LinkedIn job ID from the URL (e.g. linkedin.com/jobs/view/1234567890/) and pass it as job_id.",
    {"company": str, "description": str, "rating": int, "content": str, "job_id": str},
)
async def save_job_posting(args: dict[str, Any]) -> dict:
    return await do_save_job_posting(
        args["company"], args["description"], args["rating"], args["content"],
        job_id=args.get("job_id"),
    )


@tool(
    "notify_user",
    "Send a Telegram notification to the user. Use for jobs rated 4 or 5.",
    {"message": str},
)
async def notify_user(args: dict[str, Any]) -> dict:
    return await do_notify_user(args["message"])


@tool(
    "update_job_requirements",
    "Rewrite JOB_REQUIREMENTS.md with a complete, updated summary of the user's job preferences. "
    "Always rewrite the full file — never append. Returns the new contents so they are in context.",
    {"content": str},
)
async def update_job_requirements(args: dict[str, Any]) -> dict:
    return await do_update_job_requirements(args["content"])


@tool(
    "check_and_record_job",
    "Before evaluating any job, call this with the site name, job ID, company name, and job title/description. "
    "Returns 'already_processed' (skip it), 'too_old' (skip it), or 'new' (proceed to evaluate). "
    "date_posted is optional — pass whatever is visible (YYYY-MM-DD or relative like '4 days ago'); omit if not shown. "
    "Optionally pass url (the job posting URL) and content (full text of the posting) to persist them in the record.",
    {"site": str, "job_id": str, "company": str, "description": str, "date_posted": str, "url": str, "content": str},
)
async def check_and_record_job(args: dict[str, Any]) -> dict:
    return await do_check_and_record_job(
        args["site"], args["job_id"], args["company"], args["description"],
        date_posted=args.get("date_posted"), url=args.get("url"), content=args.get("content"),
    )


@tool(
    "queue_candidate",
    "Add a job candidate to the internal evaluation queue. "
    "Call this after check_and_record_job returns 'new'. "
    "Pass what is visible in the search results: URL, title, company, snippet. "
    "date_posted is optional — pass it if visible (exact or relative), omit if not shown. "
    "Do NOT navigate to the individual job page — a separate agent handles that in stage 2.",
    {"site": str, "job_id": str, "url": str, "title": str, "company": str, "snippet": str, "date_posted": str},
)
async def queue_candidate(args: dict[str, Any]) -> dict:
    return await do_queue_candidate(
        args["site"], args["job_id"], args["url"], args["title"],
        args["company"], args["snippet"], date_posted=args.get("date_posted"),
    )


def make_job_search_server(interactive: bool):
    """MCP server for interactive mode."""
    tools = [check_and_record_job, save_job_posting, notify_user]
    if interactive:
        tools.append(update_job_requirements)
    return create_sdk_mcp_server(name="job_search", version="1.0.0", tools=tools)


def make_scraper_server():
    """MCP server for stage 1: collects candidates from search results."""
    return create_sdk_mcp_server(
        name="job_scraper", version="1.0.0",
        tools=[check_and_record_job, queue_candidate],
    )


def make_evaluator_server():
    """MCP server for stage 2: saves evaluated jobs and sends notifications."""
    return create_sdk_mcp_server(
        name="job_evaluator", version="1.0.0",
        tools=[save_job_posting, notify_user],
    )


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
    matches = list(PROJECT_DIR.glob("*-resume-*.md")) + list(PROJECT_DIR.glob("*-resume-*.pdf"))
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
        "Based on the resume and job requirements above, generate 3–4 short LinkedIn job search queries "
        "(2–6 words each, like a job title) that will surface the most relevant senior AI/ML roles. "
        'Reply with ONLY a JSON array of strings, e.g. ["Principal AI Engineer", "Staff ML Engineer"].'
    )
    options = ClaudeAgentOptions(
        model="claude-sonnet-4-6",
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


# --- Main ---

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
    jobs_after = set(PROJECT_DIR.glob("saved_jobs-*/job_posting-*.md"))
    new_jobs = jobs_after - jobs_before
    num_evaluated = len(new_jobs)
    num_high_rated = sum(
        1 for f in new_jobs
        if (m := _RATING_RE.search(f.name)) and int(m.group(1)) >= 4
    )
    return num_evaluated, num_high_rated


async def run_scraper(client: ClaudeSDKClient, queries: list[str]) -> None:
    """Stage 1: haiku scraper collects candidates from LinkedIn search results."""
    query_list = "\n".join(f'- "{q}"' for q in queries)
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
                    print(block.text, end="", flush=True)
        elif isinstance(msg, ResultMessage):
            print()
            print_result_stats(msg)


async def evaluate_candidate(candidate: dict, playwright_mcp: dict, evaluator_prompt: str) -> None:
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


async def run_non_interactive() -> None:
    global _candidates
    _candidates = []

    console.print("[bold cyan]Job Search Agent — Non-interactive Mode[/bold cyan]")
    console.print("[cyan]" + "=" * 40 + "[/cyan]")

    start_time = time.time()
    jobs_before = set(PROJECT_DIR.glob("saved_jobs-*/job_posting-*.md"))

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
        # model="claude-haiku-4-5-20251001",
        model="claude-haiku-4-5",
        # thinking={"type": "disabled"},
    )
    async with ClaudeSDKClient(scraper_options) as scraper:
        await run_scraper(scraper, queries)

    console.print(f"\n[dim]Stage 1 complete: {len(_candidates)} candidate(s) queued.[/dim]\n")

    if not _candidates:
        console.print("[dim]No new candidates found.[/dim]")
        elapsed_mins = (time.time() - start_time) / 60
        send_telegram(f"Job search run complete\n• No new candidates\n• Elapsed: {elapsed_mins:.1f} min")
        return

    # Stage 2: sonnet evaluator — one fresh session per job, system prompt cached after job 1
    console.print("[yellow]Stage 2: Evaluating candidates ...[/yellow]\n")
    evaluator_prompt = build_evaluator_prompt()  # built once, reused for all jobs
    for candidate in _candidates:
        await evaluate_candidate(candidate, playwright_mcp, evaluator_prompt)

    elapsed_mins = (time.time() - start_time) / 60
    num_evaluated, num_high_rated = count_new_jobs(jobs_before)

    stats_lines = [
        "Job search run complete",
        f"• Candidates found: {len(_candidates)}",
        f"• Jobs saved: {num_evaluated}",
        f"• Jobs rated ≥4: {num_high_rated}",
        f"• Elapsed: {elapsed_mins:.1f} min",
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


if __name__ == "__main__":
    asyncio.run(main())
