
import json
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pypdf
import yaml
from rich.console import Console

from agentic_job_search.config import JOB_MAX_AGE_DAYS, MODEL_NAME_LOW
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ResultMessage,
    create_sdk_mcp_server,
    query as sdk_query,
    tool,
)

console = Console()

PROJECT_DIR = Path(__file__).parent.parent.parent  # src/agentic_job_search/ -> src/ -> project root
RUN_DIR = PROJECT_DIR / "run_dir"
JOB_REQUIREMENTS_PATH = RUN_DIR / "JOB_REQUIREMENTS.md"
PROCESSED_JOBS_DIR = RUN_DIR / "processed_jobs"

_processed_jobs: set[tuple[str, str]] = set()
_candidates: list[dict] = []
_candidates_per_query: dict[str, int] = {}
_applied_companies: dict[str, str] = {}  # company_name -> PDF filename
_reference_job_texts: list[str] = []     # extracted text for evaluator prompt injection
_job_extracts: list[dict] = []           # condensed page extracts captured from the Stage 2 extractor


async def _extract_company_from_text(text: str) -> str:
    """Use agent SDK to extract the hiring company name from job description text."""
    options = ClaudeAgentOptions(
        model=MODEL_NAME_LOW,
        tools=[],
        permission_mode='bypassPermissions',
        output_format={
            'type': 'json_schema',
            'schema': {
                'type': 'object',
                'properties': {
                    'company_name': {'type': 'string', 'description': 'Name of the hiring company'},
                },
                'required': ['company_name'],
            },
        },
        cwd=str(PROJECT_DIR),
    )
    async for msg in sdk_query(prompt=f'What company posted this job?\n\n{text[:3000]}', options=options):
        if isinstance(msg, ResultMessage) and msg.structured_output:
            return msg.structured_output.get('company_name', '')
    return ''


async def company_matches_applied(candidate: str) -> str | None:
    """Return the source PDF filename if candidate matches a previously applied-to company, else None."""
    if not _applied_companies:
        return None

    companies_list = '\n'.join(f'- {name}' for name in _applied_companies)
    options = ClaudeAgentOptions(
        model=MODEL_NAME_LOW,
        tools=[],
        permission_mode='bypassPermissions',
        output_format={
            'type': 'json_schema',
            'schema': {
                'type': 'object',
                'properties': {
                    'matches': {'type': 'boolean', 'description': 'True if same organization'},
                    'matched_company_name': {'type': 'string', 'description': 'Matching name from the list, or empty string'},
                },
                'required': ['matches', 'matched_company_name'],
            },
        },
        cwd=str(PROJECT_DIR),
    )
    structured: dict | None = None
    async for msg in sdk_query(
        prompt=f'Does "{candidate}" refer to the same organization as any of these companies?\n\n{companies_list}',
        options=options,
    ):
        if isinstance(msg, ResultMessage) and msg.structured_output:
            structured = msg.structured_output
    if not structured or not structured.get('matches'):
        return None
    matched = structured.get('matched_company_name', '')
    return _applied_companies.get(matched, '<unknown PDF>')


async def _categorize_pdf_text(text: str, filename: str) -> str:
    """Use agent SDK to assign a category label to a PDF. Returns a snake_case string."""
    options = ClaudeAgentOptions(
        model=MODEL_NAME_LOW,
        tools=[],
        permission_mode='bypassPermissions',
        output_format={
            'type': 'json_schema',
            'schema': {
                'type': 'object',
                'properties': {
                    'category': {
                        'type': 'string',
                        'description': (
                            'Category for the PDF. Use "saved_jd" for job descriptions/postings, '
                            '"other" for unrelated content, or a short snake_case label you derive '
                            'such as "resume", "contract", "invoice", "article".'
                        ),
                    },
                },
                'required': ['category'],
            },
        },
        cwd=str(PROJECT_DIR),
    )
    prompt = f'Filename: {filename}\n\nContent:\n{text}'
    category: str | None = None
    async for msg in sdk_query(prompt=prompt, options=options):
        if isinstance(msg, ResultMessage) and msg.structured_output:
            category = msg.structured_output.get('category', '')
    if category is not None:
        return underscorify(category) or 'other'
    console.print(f'[yellow]Warning: no structured output for {filename}[/yellow]')
    return None


async def categorize_downloads_pdfs(downloads_dir: Path | None = None) -> None:
    """Rename uncategorized PDFs in ~/Downloads with a cat-<category>- prefix."""
    downloads = downloads_dir or Path.home() / 'Downloads'
    uncategorized = [p for p in downloads.glob('*.pdf') if not p.name.startswith('cat-')]
    if not uncategorized:
        return
    console.print(f'[dim]Categorizing {len(uncategorized)} uncategorized PDF(s) in Downloads...[/dim]')
    for pdf in uncategorized:
        try:
            reader = pypdf.PdfReader(pdf)
            text = '\n'.join(page.extract_text() or '' for page in reader.pages)
        except Exception as e:
            console.print(f'[yellow]Warning: could not read {pdf.name}: {e}[/yellow]')
            continue
        console.print(f'[dim]  Categorizing {pdf.name}...[/dim]')
        try:
            category = await _categorize_pdf_text(text[:3000], pdf.name)
        except Exception as ex:
            console.print(f'[yellow]Warning: categorization failed for {pdf.name}, leaving uncategorized: {ex}[/yellow]')
            continue
        if category is None:
            continue
        new_name = f'cat-{category}-{pdf.name}'
        console.print(f'[dim]  → {new_name}[/dim]')
        pdf.rename(pdf.parent / new_name)


async def load_downloads_applied_pdfs(cache_path: Path | None = None) -> None:
    """Load cat-saved_jd-*.pdf from ~/Downloads, extracting company names and full text."""
    global _applied_companies, _reference_job_texts

    if cache_path is None:
        cache_path = RUN_DIR / 'downloads_pdf_cache.yaml'

    downloads = Path.home() / 'Downloads'

    cache: dict = {}
    if cache_path.exists():
        try:
            cache = yaml.safe_load(cache_path.read_text(encoding='utf-8')) or {}
        except Exception as e:
            console.print(f'[yellow]Warning: could not read cache {cache_path}: {e}[/yellow]')
            cache = {}

    companies: dict[str, str] = {}  # company_name -> pdf filename
    texts: list[str] = []
    cache_dirty = False

    pi = 0
    for pi, pdf in enumerate(downloads.glob('cat-saved_jd-*.pdf')):
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

        # PDF extraction can yield null bytes; they make the text unusable as a
        # CLI subprocess argument (system prompt) — strip on load, covering
        # both fresh extractions and previously cached entries.
        text = text.replace('\x00', '')

        if company:
            companies[company] = pdf.name
        if text:
            texts.append(text)

    if cache_dirty:
        cache_path.write_text(yaml.dump(cache, default_flow_style=False), encoding='utf-8')

    _applied_companies = companies
    _reference_job_texts = texts
    console.print(f'[dim]Loaded {pi} applied-job PDF(s) from Downloads ({len(companies)} companies).[/dim]')


def underscorify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


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
        except Exception as e:
            console.print(f'[yellow]Warning: could not read {f}: {e}[/yellow]')

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
    # Cap name components — a long description (e.g. a full triage-reason sentence)
    # can push the filename past the filesystem's 255-byte limit.
    company_part = underscorify(company)[:40].rstrip('_')
    description_part = underscorify(description)[:120].rstrip('_')
    filename = f"job_posting-{id_part}{rating_part}-{company_part}-{description_part}-{ts}.md"
    (dir_path / filename).write_text(content, encoding="utf-8")

    return {"content": [{"type": "text", "text": f"Saved: saved_jobs-{date_str}/{filename}"}]}


async def do_update_job_requirements(content: str) -> dict:
    JOB_REQUIREMENTS_PATH.write_text(content, encoding="utf-8")
    return {
        "content": [
            {"type": "text", "text": f"JOB_REQUIREMENTS.md updated. New contents:\n\n{content}"}
        ]
    }


COST_LOG_PATH = RUN_DIR / "cost_log.jsonl"


def log_run_cost(record: dict, log_path: Path | None = None) -> None:
    """Append a per-run cost record as one JSON line to cost_log.jsonl."""
    path = log_path if log_path is not None else COST_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


_AUTH_REQUIRED_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'must be (legally )?authorized to work in the (us|united states)',
        r'no visa sponsorship',
        r'not able to (provide |offer )?sponsor',
        r'cannot (provide |offer )?sponsor',
        r'sponsorship (is )?not (available|provided|offered)',
        r'will not (provide |offer )?sponsor',
        r'unable to (provide |offer )?sponsor',
        r'us citizens? and (lawful )?permanent residents?',
        r'currently (legally )?authorized to work in the (us|united states)',
        r"this (position|role|job) (does not|doesn't) (provide|offer|support) (visa )?sponsorship",
        r'employment authorization (without|not requiring) sponsorship',
    ]
]


def _requires_current_us_auth(text: str) -> bool:
    """Return True if text indicates the job requires current US work authorization (no sponsorship)."""
    return any(p.search(text) for p in _AUTH_REQUIRED_PATTERNS)


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

    combined_text = ' '.join(filter(None, [description, content]))
    if _requires_current_us_auth(combined_text):
        console.print(f'[dim]Skipping {company} — requires current US work authorization.[/dim]')
        return {"content": [{"type": "text", "text": "auth_required"}]}

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


async def do_submit_job_extract(
    title: str, company: str, description: str,
    location: str | None = None, date_posted: str | None = None,
    closed: bool = False, salary: str | None = None, sponsorship_note: str | None = None,
    language_requirement: str | None = None, relocation: str | None = None,
) -> dict:
    _job_extracts.append({
        'title': title, 'company': company, 'description': description,
        'location': location or '', 'date_posted': date_posted or '',
        'closed': closed, 'salary': salary or '', 'sponsorship_note': sponsorship_note or '',
        'language_requirement': language_requirement or '', 'relocation': relocation or '',
    })
    return {"content": [{"type": "text", "text": "Extract submitted."}]}


async def do_queue_candidate(
    site: str, job_id: str, url: str, title: str, company: str,
    snippet: str, date_posted: str | None = None, query: str | None = None,
) -> dict:
    _candidates.append({
        "site": site, "job_id": job_id, "url": url,
        "title": title, "company": company,
        "date_posted": date_posted or "", "snippet": snippet,
    })
    if query:
        _candidates_per_query[query] = _candidates_per_query.get(query, 0) + 1
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
    "Returns 'already_processed' (skip it), 'too_old' (skip it), 'already_applied' (skip it), 'auth_required' (skip it — requires current US work authorization), or 'new' (proceed to evaluate). "
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
    "query is optional — pass the search query string that returned this result (e.g. 'Staff ML Engineer'). "
    "Do NOT navigate to the individual job page — a separate agent handles that in stage 2.",
    {"site": str, "job_id": str, "url": str, "title": str, "company": str, "snippet": str, "date_posted": str, "query": str},
)
async def queue_candidate(args: dict[str, Any]) -> dict:
    return await do_queue_candidate(
        args["site"], args["job_id"], args["url"], args["title"],
        args["company"], args["snippet"], date_posted=args.get("date_posted"),
        query=args.get("query"),
    )


@tool(
    "submit_job_extract",
    "Submit the condensed extract of the job posting page you navigated to. "
    "Include only information-dense content: requirements, responsibilities, stack, seniority — "
    "strip navigation chrome, boilerplate, and similar-jobs lists. "
    "Set closed=true if the page shows 'No longer accepting applications' or equivalent. "
    "Pass date_posted exactly as shown (absolute or relative like '4 days ago'). "
    "Pass sponsorship_note with any visa/work-authorization statement, verbatim, if present. "
    "Pass language_requirement with languages explicitly REQUIRED (not nice-to-have), comma-separated "
    "lowercase, e.g. 'english, german'; omit if no language requirement is stated. "
    "Pass relocation with the country/city if the posting requires relocating to or residing in a "
    "specific place (e.g. 'must be based in Portugal'); omit for work-from-anywhere roles.",
    {
        'type': 'object',
        'properties': {
            'title': {'type': 'string'},
            'company': {'type': 'string'},
            'description': {'type': 'string', 'description': 'Condensed posting content (requirements, responsibilities, stack, seniority)'},
            'location': {'type': 'string'},
            'date_posted': {'type': 'string'},
            'closed': {'type': 'boolean'},
            'salary': {'type': 'string'},
            'sponsorship_note': {'type': 'string'},
            'language_requirement': {'type': 'string', 'description': "Explicitly required languages, comma-separated lowercase, e.g. 'english, german'"},
            'relocation': {'type': 'string', 'description': 'Location the candidate must relocate to / reside in, if the posting requires one'},
        },
        'required': ['title', 'company', 'description'],
    },
)
async def submit_job_extract(args: dict[str, Any]) -> dict:
    return await do_submit_job_extract(
        args["title"], args["company"], args["description"],
        location=args.get("location"), date_posted=args.get("date_posted"),
        closed=args.get("closed", False), salary=args.get("salary"),
        sponsorship_note=args.get("sponsorship_note"),
        language_requirement=args.get("language_requirement"), relocation=args.get("relocation"),
    )


# --- MCP server factories ---

def make_job_search_server(interactive: bool):
    """MCP server for interactive mode."""
    tools = [check_and_record_job, save_job_posting]
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
    """MCP server for stage 2: captures the condensed page extract from the Haiku extractor."""
    return create_sdk_mcp_server(
        name="job_evaluator", version="1.0.0",
        tools=[submit_job_extract],
    )
