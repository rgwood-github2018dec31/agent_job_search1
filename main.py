import argparse
import asyncio
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, datetime
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


async def do_check_and_record_job(
    site: str, job_id: str, date_posted: str, company: str, description: str,
    url: str | None = None, content: str | None = None
) -> dict:
    key = (site, job_id)
    if key in _processed_jobs:
        return {"content": [{"type": "text", "text": "already_processed"}]}

    try:
        posted = date.fromisoformat(date_posted)
        if (date.today() - posted).days > 21:
            return {"content": [{"type": "text", "text": "too_old"}]}
    except ValueError:
        pass  # unparseable date: proceed anyway

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
    "Before evaluating any job, call this with the site name, job ID, posting date (YYYY-MM-DD), "
    "company name, and job title/description. Returns 'already_processed' (skip it), "
    "'too_old' (skip it), or 'new' (proceed to evaluate). "
    "Optionally pass url (the job posting URL) and content (full text of the posting) to persist them in the record.",
    {"site": str, "job_id": str, "date_posted": str, "company": str, "description": str, "url": str, "content": str},
)
async def check_and_record_job(args: dict[str, Any]) -> dict:
    return await do_check_and_record_job(
        args["site"], args["job_id"], args["date_posted"], args["company"], args["description"],
        url=args.get("url"), content=args.get("content"),
    )


def make_job_search_server(interactive: bool):
    tools = [check_and_record_job, save_job_posting, notify_user]
    if interactive:
        tools.append(update_job_requirements)
    return create_sdk_mcp_server(
        name="job_search",
        version="1.0.0",
        tools=tools,
    )


# --- Prompt construction ---

AGENT_INSTRUCTIONS_COMMON = """You are a personal job search agent.

Do not write files directly to disk. Use the save_job_posting tool to persist job data — it is the only way you should store anything.

When browsing LinkedIn:
- Navigate to https://www.linkedin.com/jobs/ to search for jobs
- Extract: job title, company, location, salary (if shown), and key requirements
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

AGENT_INSTRUCTIONS_NON_INTERACTIVE = AGENT_INSTRUCTIONS_COMMON + """
## Non-interactive mode

You are running autonomously without a user present. Your job is to:
1. Search LinkedIn for new job postings matching the requirements
2. Evaluate and rate each job
3. Save every job with save_job_posting
4. Notify the user (via notify_user) for any job rated 4 or 5

Do not ask for user input or feedback. Complete the search autonomously and then stop.
Do not update JOB_REQUIREMENTS.md.
"""


def load_resume() -> str | None:
    matches = list(PROJECT_DIR.glob("R_Garth_Wood-resume-*.*"))
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
    # Static content first (maximises cache-prefix hits across runs), dynamic content last.
    parts = []

    if interactive:
        # Resume only needed in interactive mode; non-interactive uses JOB_REQUIREMENTS.md criteria.
        resume = load_resume()
        if resume:
            parts.append(f"--- RESUME ---\n{resume}\n--- END RESUME ---")
        else:
            console.print("[yellow]Warning: no resume file found matching R_Garth_Wood-resume-*.*[/yellow]")

    if JOB_REQUIREMENTS_PATH.exists():
        requirements = JOB_REQUIREMENTS_PATH.read_text(encoding="utf-8")
        parts.append(f"--- JOB_REQUIREMENTS.md ---\n{requirements}\n--- END JOB_REQUIREMENTS.md ---")

    instructions = AGENT_INSTRUCTIONS_INTERACTIVE if interactive else AGENT_INSTRUCTIONS_NON_INTERACTIVE
    parts.append(instructions)

    # Dynamic content last so the static prefix above can be cached.
    parts.append(f"Today's date: {datetime.now().strftime('%Y-%m-%d')}.")

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


async def run_non_interactive(client: ClaudeSDKClient) -> None:
    console.print("[bold cyan]Job Search Agent — Non-interactive Mode[/bold cyan]")
    console.print("[cyan]" + "=" * 40 + "[/cyan]")
    console.print("[yellow]Searching LinkedIn for new jobs - running autonomously ...[/yellow]\n")

    start_time = time.time()
    jobs_before = set(PROJECT_DIR.glob("saved_jobs-*/job_posting-*.md"))

    initial = (
        "Search LinkedIn for new job postings that match the resume and requirements. "
        "Evaluate each job, save it with save_job_posting (all ratings), and call notify_user "
        "for any job rated 4 or 5. Work autonomously without asking for input."
    )
    await client.query(initial)

    total_cost: float | None = None
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
            total_cost = msg.total_cost_usd

    elapsed_mins = (time.time() - start_time) / 60
    num_evaluated, num_high_rated = count_new_jobs(jobs_before)

    stats_lines = [
        "Job search run complete",
        f"• Jobs evaluated: {num_evaluated}",
        f"• Jobs rated ≥4: {num_high_rated}",
        f"• Cost: ${total_cost:.4f}" if total_cost is not None else "• Cost: unknown",
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
    system_prompt = build_system_prompt(interactive)

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        mcp_servers={
            "playwright": {
                "type": "stdio",
                "command": "npx",
                "args": ["@playwright/mcp@latest", "--user-data-dir", str(BROWSER_PROFILE_DIR)],
            },
            "job_search": make_job_search_server(interactive),
        },
        permission_mode="acceptEdits",
        cwd=str(PROJECT_DIR),
        effort="low" if not interactive else None,
    )

    async with ClaudeSDKClient(options) as client:
        if interactive:
            await run_interactive(client)
        else:
            await run_non_interactive(client)


if __name__ == "__main__":
    asyncio.run(main())
