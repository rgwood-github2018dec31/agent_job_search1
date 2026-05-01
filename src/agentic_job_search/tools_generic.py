import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pypdf
import requests
import yaml
from rich.console import Console

from agentic_job_search.config import JOB_MAX_AGE_DAYS, JOB_SEARCH_START_DATE, MODEL_NAME_LOW
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    create_sdk_mcp_server,
    tool,
)

console = Console()

PROJECT_DIR = Path(__file__).parent.parent.parent  # src/agentic_job_search/ -> src/ -> project root
RUN_DIR = PROJECT_DIR / "run_dir"
JOB_REQUIREMENTS_PATH = RUN_DIR / "JOB_REQUIREMENTS.md"
PROCESSED_JOBS_DIR = RUN_DIR / "processed_jobs"

_processed_jobs: set[tuple[str, str]] = set()
_candidates: list[dict] = []
_applied_companies: dict[str, str] = {}  # company_name -> PDF filename
_reference_job_texts: list[str] = []     # extracted text for evaluator prompt injection


async def _extract_company_from_text(text: str) -> str:
    """Use agent SDK to extract the hiring company name from job description text."""
    captured: list[str] = []

    @tool('record_company', 'Record the company that posted this job', {
        'type': 'object',
        'properties': {
            'company_name': {'type': 'string', 'description': 'Name of the hiring company'},
        },
        'required': ['company_name'],
    })
    async def _record(args: dict[str, Any]) -> dict:
        captured.append(args['company_name'])
        return {'content': [{'type': 'text', 'text': 'Recorded.'}]}

    server = create_sdk_mcp_server(name='company_extractor', version='1.0.0', tools=[_record])
    options = ClaudeAgentOptions(
        model=MODEL_NAME_LOW,
        mcp_servers={'company_extractor': server},
        permission_mode='acceptEdits',
        cwd=str(PROJECT_DIR),
    )
    async with ClaudeSDKClient(options) as client:
        await client.query(f'What company posted this job? Call record_company with the result.\n\n{text[:3000]}')
    return captured[0] if captured else ''


async def company_matches_applied(candidate: str) -> str | None:
    """Return the source PDF filename if candidate matches a previously applied-to company, else None."""
    if not _applied_companies:
        return None

    captured: list[dict] = []
    companies_list = '\n'.join(f'- {name}' for name in _applied_companies)

    @tool('record_match_result', 'Record whether the candidate matches an applied company', {
        'type': 'object',
        'properties': {
            'matches': {'type': 'boolean', 'description': 'True if same organization'},
            'matched_company_name': {'type': 'string', 'description': 'Matching name from the list, or empty string'},
        },
        'required': ['matches', 'matched_company_name'],
    })
    async def _record(args: dict[str, Any]) -> dict:
        captured.append(args)
        return {'content': [{'type': 'text', 'text': 'Recorded.'}]}

    server = create_sdk_mcp_server(name='company_matcher', version='1.0.0', tools=[_record])
    options = ClaudeAgentOptions(
        model=MODEL_NAME_LOW,
        mcp_servers={'company_matcher': server},
        permission_mode='acceptEdits',
        cwd=str(PROJECT_DIR),
    )
    async with ClaudeSDKClient(options) as client:
        await client.query(
            f'Does "{candidate}" refer to the same organization as any of these companies?\n\n'
            f'{companies_list}\n\nCall record_match_result with your answer.'
        )
    if not captured:
        return None
    result = captured[0]
    if not result['matches']:
        return None
    matched = result.get('matched_company_name', '')
    return _applied_companies.get(matched, '<unknown PDF>')


async def load_downloads_applied_pdfs(cache_path: Path | None = None) -> None:
    """Scan ~/Downloads for job description PDFs saved April 2026+, extract companies and text."""
    global _applied_companies, _reference_job_texts

    if cache_path is None:
        cache_path = RUN_DIR / 'downloads_pdf_cache.yaml'

    cutoff = JOB_SEARCH_START_DATE.timestamp()
    downloads = Path.home() / 'Downloads'

    cache: dict = {}
    if cache_path.exists():
        try:
            cache = yaml.safe_load(cache_path.read_text(encoding='utf-8')) or {}
        except Exception:
            cache = {}

    pdfs = [p for p in downloads.glob('*.pdf') if p.stat().st_mtime >= cutoff]

    companies: dict[str, str] = {}  # company_name -> pdf filename
    texts: list[str] = []
    cache_dirty = False

    for pdf in pdfs:
        key = str(pdf)
        mtime = pdf.stat().st_mtime
        entry = cache.get(key)

        if entry and abs(entry.get('mtime', 0) - mtime) < 1.0:
            company = entry.get('company', '')
            text = entry.get('text', '')
        else:
            try:
                reader = pypdf.PdfReader(pdf)
                text = '\n'.join(page.extract_text() or '' for page in reader.pages)
            except Exception as e:
                console.print(f'[yellow]Warning: could not read {pdf.name}: {e}[/yellow]')
                continue

            company = await _extract_company_from_text(text)
            cache[key] = {'mtime': mtime, 'company': company, 'text': text}
            cache_dirty = True

        if company:
            companies[company] = pdf.name
        if text:
            texts.append(text)

    if cache_dirty:
        cache_path.write_text(yaml.dump(cache, default_flow_style=False), encoding='utf-8')

    _applied_companies = companies
    _reference_job_texts = texts
    console.print(f'[dim]Loaded {len(pdfs)} applied-job PDF(s) from Downloads ({len(companies)} companies).[/dim]')


def underscorify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def send_telegram(text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        console.print("[yellow]Warning: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set, skipping notification.[/yellow]")
        return
    console.print(f'[dim]→ POST api.telegram.org/sendMessage[/dim]')
    requests.post(f'https://api.telegram.org/bot{token}/sendMessage', data={'chat_id': chat_id, 'text': text})


# Old MD filename pattern: job_posting-{linkedin_id}[-rating_{N}]-{company}-{desc}-{timestamp}.md
_SAVED_JOB_MD_RE = re.compile(r"^job_posting-(\d+|noid)(?:-rating_\d+)?-.+-\d+\.md$")


def load_processed_jobs() -> None:
    """Populate _processed_jobs from both old MD filenames and new per-job YAML files."""
    global _processed_jobs
    ids: set[tuple[str, str]] = set()

    # Old format: extract (linkedin, job_id) from saved MD filenames
    for f in RUN_DIR.glob("saved_jobs-*/job_posting-*.md"):
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
    dir_path = RUN_DIR / f"saved_jobs-{date_str}"
    dir_path.mkdir(parents=True, exist_ok=True)

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

    matched_pdf = await company_matches_applied(company)
    if matched_pdf is not None:
        console.print(f"[dim]Skipping {company} — already applied ({matched_pdf}).[/dim]")
        return {"content": [{"type": "text", "text": "already_applied"}]}

    posted = parse_posting_date(date_posted)
    if posted and (date.today() - posted).days > JOB_MAX_AGE_DAYS:
        return {"content": [{"type": "text", "text": "too_old"}]}

    PROCESSED_JOBS_DIR.mkdir(parents=True, exist_ok=True)
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
    "Returns 'already_processed' (skip it), 'too_old' (skip it), 'already_applied' (skip it), or 'new' (proceed to evaluate). "
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


# --- MCP server factories ---

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
