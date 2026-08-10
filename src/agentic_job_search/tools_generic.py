
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pypdf
import yaml
from rich.console import Console

from agentic_job_search.config import (
    APPLIED_JOBS_HORIZON_DAYS,
    COMPANY_MATCH_PROVIDER,
    JOB_MAX_AGE_DAYS,
    MODEL_NAME_LOW,
)
from agentic_job_search.triage import chat_openrouter, extract_json_object
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
APPLIED_JOBS_DIR = RUN_DIR / "applied_jobs"
APPLIED_JOBS_INDEX_PATH = APPLIED_JOBS_DIR / "index.yaml"
# Legacy per-path cache from when the corpus lived in ~/Downloads; still read during ingest
# as a fallback source of applied dates if a file's mtime has drifted.
LEGACY_DOWNLOADS_CACHE_PATH = RUN_DIR / "downloads_pdf_cache.yaml"

# Leading wildcard so an already-date-prefixed PDF landing back in Downloads is still ingested
# (with its original date) rather than silently ignored.
APPLIED_PDF_GLOB = '*cat-saved_jd-*.pdf'
_DATE_PREFIX_RE = re.compile(r'^(\d{4}-\d{2}-\d{2})-')

logger = logging.getLogger(__name__)

_processed_jobs: set[tuple[str, str]] = set()
_candidates: list[dict] = []
_candidates_per_query: dict[str, int] = {}
_applied_companies: dict[str, str] = {}  # company_name -> PDF filename
_reference_job_texts: list[str] = []     # extracted text for evaluator prompt injection
# In-horizon applied-job records: filename, applied_date, company, job_title, is_agency, end_client
_applied_jobs: list[dict] = []
_job_extracts: list[dict] = []           # condensed page extracts captured from the Stage 2 extractor
# Scraper funnel observability: counts every check_and_record_job outcome this run.
_check_status_counts: dict[str, int] = {}  # status -> count (new/already_processed/already_applied/too_old/auth_required)
# Per-listing audit trail: one record per check_and_record_job call, keyed by (site, job_id).
# Feeds the per-run audit log and the --audit-opus un-surfaced sampling.
_listing_records: dict[tuple[str, str], dict] = {}
# query -> listings inspected, so "never searched" is distinguishable from "searched, found nothing"
_queries_searched: dict[str, Any] = {}


async def _extract_applied_job_metadata(text: str, filename: str) -> dict:
    """Extract company, title, and recruiter-agency signal from an applied-job PDF's text.

    Recruiting agencies and job aggregators repost roles on behalf of other companies, so the
    name on the posting is often not the employer. Blocklisting such a name would suppress every
    future listing that agency posts, hence the explicit is_recruiting_agency / end_client_name.
    """
    options = ClaudeAgentOptions(
        model=MODEL_NAME_LOW,
        tools=[],
        permission_mode='bypassPermissions',
        output_format={
            'type': 'json_schema',
            'schema': {
                'type': 'object',
                'properties': {
                    'company_name': {'type': 'string', 'description': 'Name of the organization on the posting'},
                    'job_title': {'type': 'string', 'description': 'Job title of the role'},
                    'is_recruiting_agency': {
                        'type': 'boolean',
                        'description': (
                            'True if the poster is a staffing firm, recruiting agency, or job aggregator '
                            'reposting on behalf of another company, rather than the employer that '
                            'would actually hire.'
                        ),
                    },
                    'end_client_name': {
                        'type': 'string',
                        'description': 'The company that would actually hire, if the posting names one; otherwise an empty string',
                    },
                },
                'required': ['company_name', 'job_title', 'is_recruiting_agency', 'end_client_name'],
            },
        },
        cwd=str(PROJECT_DIR),
    )
    prompt = (
        f'Filename: {filename}\n\n'
        f'Job posting text:\n{text[:3000]}\n\n'
        'Identify the organization posting this job, the job title, whether the poster is a '
        'recruiting agency/aggregator rather than the hiring employer, and the end client if named.'
    )
    async for msg in sdk_query(prompt=prompt, options=options):
        if isinstance(msg, ResultMessage) and msg.structured_output:
            out = msg.structured_output
            return {
                'company': out.get('company_name', ''),
                'job_title': out.get('job_title', ''),
                'is_agency': bool(out.get('is_recruiting_agency', False)),
                'end_client': out.get('end_client_name', ''),
            }
    console.print(f'[yellow]Warning: no structured output extracting metadata for {filename}[/yellow]')
    return {'company': '', 'job_title': '', 'is_agency': False, 'end_client': ''}


def _normalize_company(name: str) -> str:
    """Strip punctuation and common legal/entity suffixes so 'Datadog, Inc.' == 'Datadog'."""
    cleaned = re.sub(r'[^a-z0-9 ]+', ' ', name.lower())
    words = [w for w in cleaned.split() if w not in _COMPANY_NOISE_WORDS]
    return ' '.join(words)


_COMPANY_NOISE_WORDS = {
    'inc', 'llc', 'ltd', 'limited', 'corp', 'corporation', 'co', 'company',
    'group', 'holdings', 'gmbh', 'sa', 'bv', 'plc', 'the', 'technologies', 'software',
}


async def _company_matches_openrouter(candidate: str, companies_list: str) -> dict:
    """Fuzzy company match via the OpenRouter MCP server (glm). Raises on any failure."""
    content, _cost = await chat_openrouter(
        f'Does "{candidate}" refer to the same organization as any of these companies?\n\n'
        f'{companies_list}\n\n'
        'Respond with ONLY a JSON object: '
        '{"matches": <true|false>, "matched_company_name": "<exact name from the list, or empty string>"}'
    )
    return extract_json_object(content)


async def company_matches_applied(candidate: str) -> str | None:
    """Return the source PDF filename if candidate matches a previously applied-to company, else None.

    Tries an exact normalized string match first — that resolves the overwhelming majority of
    listings for free and, more importantly, without a per-listing network round-trip that would
    otherwise dominate Stage 1b wall-clock. The LLM is only consulted for genuine ambiguity.
    """
    if not _applied_companies:
        return None

    normalized = {_normalize_company(name): name for name in _applied_companies}
    exact = normalized.get(_normalize_company(candidate))
    if exact:
        logger.info(f'company_matches_applied: exact normalized match {candidate!r} -> {exact!r} (no LLM call)')
        return _applied_companies[exact]

    companies_list = '\n'.join(f'- {name}' for name in _applied_companies)

    if COMPANY_MATCH_PROVIDER == 'openrouter':
        try:
            structured = await _company_matches_openrouter(candidate, companies_list)
            if not structured.get('matches'):
                return None
            return _applied_companies.get(structured.get('matched_company_name', ''), '<unknown PDF>')
        except Exception as ex:
            logger.warning(f'company_matches_applied via OpenRouter failed, falling back to Anthropic: {ex}')

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


def _read_yaml_mapping(path: Path) -> dict:
    """Read a YAML file into a dict, warning and returning {} if it is missing or unreadable."""
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    except Exception as ex:
        console.print(f'[yellow]Warning: could not read {path}: {ex}[/yellow]')
        return {}


def _resolve_applied_date(pdf: Path, index: dict, legacy_cache: dict) -> date:
    """Determine the date a job was applied to, most durable source first.

    mtime is the least trustworthy carrier — a move, copy, or backup restore rewrites it — so
    the filename prefix and the index take precedence over it.
    """
    prefix_match = _DATE_PREFIX_RE.match(pdf.name)
    if prefix_match:
        try:
            return datetime.strptime(prefix_match.group(1), '%Y-%m-%d').date()
        except ValueError as ex:
            console.print(f'[yellow]Warning: bad date prefix on {pdf.name}: {ex}[/yellow]')

    indexed = index.get(pdf.name, {}).get('applied_date')
    if isinstance(indexed, date):
        return indexed
    if isinstance(indexed, str):
        try:
            return datetime.strptime(indexed, '%Y-%m-%d').date()
        except ValueError as ex:
            console.print(f'[yellow]Warning: bad applied_date for {pdf.name} in index: {ex}[/yellow]')

    legacy_mtime = legacy_cache.get(str(pdf), {}).get('mtime')
    if legacy_mtime:
        return datetime.fromtimestamp(legacy_mtime).date()

    return datetime.fromtimestamp(pdf.stat().st_mtime).date()


async def ingest_downloads_applied_pdfs(
    downloads_dir: Path | None = None,
    applied_dir: Path | None = None,
    dry_run: bool = False,
) -> int:
    """Move cat-saved_jd-*.pdf out of Downloads into the applied-jobs corpus, date-stamped.

    Files are renamed to '{YYYY-MM-DD}-{original name}' so the applied date survives any later
    copy that drops metadata. mtime is restored on the destination as a secondary carrier.
    Already-prefixed files are a no-op, so this is safe to run on every startup and doubles as
    the one-time migration. Returns the number of files moved (or that would move, if dry_run).
    """
    downloads = downloads_dir or Path.home() / 'Downloads'
    target_dir = applied_dir or APPLIED_JOBS_DIR

    pdfs = sorted(downloads.glob(APPLIED_PDF_GLOB))
    if not pdfs:
        return 0

    index = _read_yaml_mapping(target_dir / 'index.yaml')
    legacy_cache = _read_yaml_mapping(LEGACY_DOWNLOADS_CACHE_PATH)

    moved = 0
    for pdf in pdfs:
        applied_date = _resolve_applied_date(pdf, index, legacy_cache)
        name = pdf.name if _DATE_PREFIX_RE.match(pdf.name) else f'{applied_date.isoformat()}-{pdf.name}'
        destination = target_dir / name

        if destination.exists():
            console.print(f'[yellow]Skipping {pdf.name} — {destination.name} already in the corpus.[/yellow]')
            continue

        if dry_run:
            console.print(f'[dim]  would move {pdf.name} → {destination.name}[/dim]')
            moved += 1
            continue

        mtime = pdf.stat().st_mtime
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(pdf), str(destination))
            os.utime(destination, (mtime, mtime))
        except Exception as ex:
            # shutil.move copies then unlinks; if the unlink failed (e.g. a read-only Downloads)
            # both copies now exist and the destination-exists check would skip this file
            # forever. Roll the destination back so the next run retries cleanly.
            if pdf.exists() and destination.exists():
                try:
                    destination.unlink()
                except Exception as cleanup_ex:
                    console.print(
                        f'[yellow]Warning: {pdf.name} was copied but not removed from '
                        f'{pdf.parent}, and the partial copy could not be rolled back: {cleanup_ex}[/yellow]'
                    )
            console.print(f'[yellow]Warning: could not move {pdf.name}: {ex}[/yellow]')
            continue
        logger.info(f'Ingested applied job: {pdf.name} → {destination.name}')
        moved += 1

    if moved:
        verb = 'would ingest' if dry_run else 'Ingested'
        console.print(f'[dim]{verb} {moved} applied-job PDF(s) into {target_dir}.[/dim]')
    return moved


async def load_applied_jobs(applied_dir: Path | None = None, index_path: Path | None = None) -> None:
    """Load the applied-jobs corpus, populating the in-horizon globals.

    index.yaml caches the extracted text and metadata per filename, keyed on mtime, so the
    metadata LLM call only runs for new or changed PDFs. Records applied to longer ago than
    APPLIED_JOBS_HORIZON_DAYS stay on disk but are not read into any global.
    """
    global _applied_companies, _reference_job_texts, _applied_jobs

    target_dir = applied_dir or APPLIED_JOBS_DIR
    index_file = index_path or (target_dir / 'index.yaml')

    index = _read_yaml_mapping(index_file)
    legacy_cache = _read_yaml_mapping(LEGACY_DOWNLOADS_CACHE_PATH)
    cutoff = date.today() - timedelta(days=APPLIED_JOBS_HORIZON_DAYS)

    companies: dict[str, str] = {}  # company_name -> pdf filename
    texts: list[str] = []
    jobs: list[dict] = []
    index_dirty = False
    total = 0
    aged_out = 0

    for pdf in sorted(target_dir.glob('*.pdf')):
        total += 1
        applied_date = _resolve_applied_date(pdf, index, legacy_cache)
        mtime = pdf.stat().st_mtime
        entry = index.get(pdf.name)

        if entry and abs(entry.get('mtime', 0) - mtime) < 1.0:
            text = entry.get('text', '')
            metadata = {
                'company': entry.get('company', ''),
                'job_title': entry.get('job_title', ''),
                'is_agency': bool(entry.get('is_agency', False)),
                'end_client': entry.get('end_client', ''),
            }
        else:
            try:
                reader = pypdf.PdfReader(pdf)
                text = '\n'.join(page.extract_text() or '' for page in reader.pages)
            except Exception as ex:
                console.print(f'[yellow]Warning: could not read {pdf.name}: {ex}[/yellow]')
                continue

            metadata = await _extract_applied_job_metadata(text, pdf.name)
            index[pdf.name] = {
                'applied_date': applied_date,
                'mtime': mtime,
                'text': text,
                **metadata,
            }
            index_dirty = True

        if applied_date < cutoff:
            aged_out += 1
            continue

        # PDF extraction can yield null bytes; they make the text unusable as a
        # CLI subprocess argument (system prompt) — strip on load, covering
        # both fresh extractions and previously cached entries.
        text = text.replace('\x00', '')

        # A recruiting agency's own name must never enter the blocklist: applying to one role
        # through an aggregator would otherwise suppress every other company it posts for.
        if metadata['company'] and not metadata['is_agency']:
            companies[metadata['company']] = pdf.name
        if metadata['end_client']:
            companies[metadata['end_client']] = pdf.name
        if text:
            texts.append(text)
        jobs.append({'filename': pdf.name, 'applied_date': applied_date, **metadata})

    if index_dirty:
        target_dir.mkdir(parents=True, exist_ok=True)
        index_file.write_text(yaml.dump(index, default_flow_style=False, allow_unicode=True), encoding='utf-8')

    _applied_companies = companies
    _reference_job_texts = texts
    _applied_jobs = jobs
    console.print(
        f'[dim]Loaded {len(jobs)} applied job(s) within {APPLIED_JOBS_HORIZON_DAYS} days '
        f'({len(companies)} blocklisted companies); {aged_out} of {total} beyond the horizon.[/dim]'
    )
    logger.info(f'Applied jobs: {len(jobs)} in horizon, {aged_out} aged out, {total} on disk')


def applied_jobs_summary() -> str:
    """One line per in-horizon applied job, for injection into the query-generation prompt."""
    lines = []
    for job in sorted(_applied_jobs, key=lambda j: j['applied_date'], reverse=True):
        title = job['job_title'] or '(unknown title)'
        company = job['company'] or '(unknown company)'
        suffix = ' [via agency]' if job['is_agency'] else ''
        if job['end_client']:
            suffix += f' [hiring company: {job["end_client"]}]'
        lines.append(f'- {title} — {company} ({job["applied_date"].isoformat()}){suffix}')
    return '\n'.join(lines)


RUN_LOCK_PATH = RUN_DIR / '.run_lock.json'


def _pid_alive(pid: int) -> bool:
    """True if a process with this PID exists. Signal 0 checks liveness without touching it."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    return True


def read_run_lock(lock_path: Path | None = None) -> dict | None:
    """Return the active run's lock record, or None if no run is currently running.

    A lock whose PID is dead is stale (the run crashed or was killed) and is reported as
    'no run active' — a leftover file must never block the next run forever.
    """
    path = lock_path or RUN_LOCK_PATH
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception as ex:
        logger.warning(f'Run lock at {path} is unreadable, treating as stale: {ex}')
        return None
    pid = data.get('pid')
    if not isinstance(pid, int) or not _pid_alive(pid):
        return None
    return data


def acquire_run_lock(mode: str, force: bool = False, lock_path: Path | None = None) -> dict | None:
    """Claim the run lock. Returns the CONFLICTING record if a run is already active, else None.

    Concurrent runs share processed_jobs/ and one Playwright browser profile: whichever run
    reaches a listing first marks it processed, so the other silently dedups it away and the
    jobs are never evaluated by either. That is invisible in the logs, which is why this is
    enforced rather than merely warned about.
    """
    path = lock_path or RUN_LOCK_PATH
    active = read_run_lock(path)
    if active and not force:
        return active
    if active and force:
        logger.warning(f'Overriding active run lock held by PID {active["pid"]} (--force)')

    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        'pid': os.getpid(),
        'started_at': datetime.now().isoformat(timespec='seconds'),
        'mode': mode,
        'argv': ' '.join(sys.argv),
    }
    path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    logger.info(f'Acquired run lock (pid {record["pid"]}, mode {mode})')
    return None


def release_run_lock(lock_path: Path | None = None) -> None:
    """Release the lock, but only if we are the holder — never delete another run's lock."""
    path = lock_path or RUN_LOCK_PATH
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        data = {}
    if data.get('pid') != os.getpid():
        logger.warning(f'Not releasing run lock at {path}: held by pid {data.get("pid")}, not us')
        return
    path.unlink()
    logger.info('Released run lock')


RUN_PROCESS_PATTERNS = ('job-search', 'main.py')


def find_run_processes() -> list[str]:
    """Scan for job-search processes that hold no lock.

    The lock only knows about runs started after it existed, and the entry point can be either
    `main.py` or the installed `job-search` console script — so the lock alone can report
    'nothing running' while a run is very much running. This is the backstop for that.
    """
    try:
        output = subprocess.run(
            ['ps', '-axo', 'pid=,command='], capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception as ex:
        logger.warning(f'Could not scan for running processes: {ex}')
        return []

    own_pid = str(os.getpid())
    matches = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        pid, _, command = line.partition(' ')
        if pid == own_pid or 'grep' in command:
            continue
        # Require an agentic-job-search entry point, not merely any file called main.py
        if any(pat in command for pat in RUN_PROCESS_PATTERNS) and 'agent_job_search' in command:
            matches.append(f'{pid}  {command[:160]}')
    return matches


def describe_run_status(lock_path: Path | None = None) -> str:
    """Human-readable answer to 'is a run going right now?'."""
    active = read_run_lock(lock_path)
    if not active:
        stray = find_run_processes()
        if stray:
            listed = '\n'.join(f'  {s}' for s in stray)
            return (
                'No run lock is held, but job-search process(es) ARE running:\n'
                f'{listed}\n'
                'These were most likely started before the run lock existed. Starting another run '
                'now would let the two steal jobs from each other via processed_jobs/.'
            )
        return 'No job search run is currently active.'
    started = active.get('started_at', '?')
    try:
        elapsed = f' ({(datetime.now() - datetime.fromisoformat(started)).total_seconds() / 60:.1f} min ago)'
    except Exception:
        elapsed = ''
    return (
        f'A run IS active:\n'
        f'  pid:     {active.get("pid")}\n'
        f'  mode:    {active.get("mode")}\n'
        f'  started: {started}{elapsed}\n'
        f'  command: {active.get("argv")}'
    )


def job_url(site: str, job_id: str) -> str:
    """Reconstruct a canonical job URL, so audit records always have one even when the
    scraper recorded a listing without passing the URL through."""
    if site == 'linkedin':
        return f'https://www.linkedin.com/jobs/view/{job_id}/'
    return f'({site} job {job_id})'


def record_job_outcome(site: str, job_id: str, outcome: str, rating: int | None = None, summary: str = '') -> None:
    """Attach a Stage 2 result to the listing's audit record."""
    record = _listing_records.get((site, job_id))
    if record is None:
        logger.warning(f'record_job_outcome: no listing record for {site}/{job_id} ({outcome})')
        return
    record['outcome'] = outcome
    record['rating'] = rating
    record['summary'] = summary


def write_run_audit_log(
    queries: list[str],
    applied_jobs_in_horizon: int,
    applied_jobs_total: int,
    funnel: dict,
    audit_findings: list[dict] | None = None,
    run_dir: Path | None = None,
    timestamp: datetime | None = None,
) -> Path:
    """Write a per-run, human-readable trace of what the pipeline actually did.

    Answers, for one run: how many applied jobs fed it, which queries ran, how many jobs each
    query surfaced, and what happened to every individual job (with URL and summary).
    """
    target_dir = (run_dir or RUN_DIR) / 'audit_logs'
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = (timestamp or datetime.now()).strftime('%Y%b%d-%H%M%S')
    path = target_dir / f'audit-{stamp}.md'

    lines = [
        f'# Run audit — {(timestamp or datetime.now()).isoformat(timespec="seconds")}',
        '',
        '## 1. Applied jobs (input signal)',
        '',
        f'- In horizon ({APPLIED_JOBS_HORIZON_DAYS} days): **{applied_jobs_in_horizon}**',
        f'- Total on disk: {applied_jobs_total}',
        f'- Blocklisted companies: {len(_applied_companies)}',
        '',
        '## 2. Search queries',
        '',
    ]

    if queries:
        lines.append(f'{len(queries)} quer(ies) generated:')
        lines.append('')
        for q in queries:
            searched = _queries_searched.get(q)
            if searched is None:
                status = '**NEVER SEARCHED**'
            elif searched == 'error':
                status = '**FAILED**'
            else:
                status = f'{searched} listing(s) inspected'
            lines.append(f'- `{q}` — {status}')
    else:
        lines.append('_No queries generated._')

    lines += ['', '## 3. Jobs found per query', '']
    if queries:
        lines += ['| Query | Listings inspected | Candidates queued |', '|---|---|---|']
        for q in queries:
            searched = _queries_searched.get(q)
            seen = 'never searched' if searched is None else searched
            lines.append(f'| {q} | {seen} | {_candidates_per_query.get(q, 0)} |')
    else:
        lines.append('_n/a_')

    lines += ['', '## 4. Per-job outcomes', '']
    if _listing_records:
        lines += ['| Company | Title | Outcome | Rating | URL | Summary |', '|---|---|---|---|---|---|']
        for record in sorted(
            _listing_records.values(),
            key=lambda r: (-(r['rating'] or 0), r['company'].lower()),
        ):
            rating = record['rating'] if record['rating'] is not None else ''
            summary = (record.get('summary') or '').replace('|', '\\|').replace('\n', ' ')[:200]
            title = (record['title'] or '').replace('|', '\\|')
            company = (record['company'] or '').replace('|', '\\|')
            lines.append(
                f'| {company} | {title} | {record["outcome"]} | {rating} | {record["url"]} | {summary} |'
            )
    else:
        lines.append('_No listings inspected._')

    lines += ['', '## 5. Funnel counters', '', '```json', json.dumps(funnel, indent=2), '```']

    if audit_findings:
        lines += ['', '## 6. Opus audit of un-surfaced jobs', '']
        lines += ['| Pool | Company | Title | Opus rating | URL | Verdict |', '|---|---|---|---|---|---|']
        for finding in audit_findings:
            lines.append(
                f'| {finding["pool"]} | {finding["company"]} | {finding["title"]} | '
                f'{finding["opus_rating"]} | {finding["url"]} | {finding["verdict"]} |'
            )
        lines += ['', 'Reasoning:', '']
        for finding in audit_findings:
            lines.append(f'- **{finding["company"]} — {finding["title"]}** ({finding["verdict"]}): {finding["reasoning"]}')

    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    logger.info(f'Run audit log written to {path}')
    return path


def unsurfaced_pools(saved_ratings: dict[tuple[str, str], int] | None = None) -> dict[str, list[dict]]:
    """Group listings that never reached a full rating, by why they were dropped.

    Feeds --audit-opus. Three pools, matching the three ways a good job can go unseen:
      - filtered: dropped by check_and_record_job (too_old / already_applied / auth_required)
      - never_queued: check said 'new' but the scraper never called queue_candidate on it
      - mid_rated: rated 2-3 by the normal rater, i.e. surfaced but not acted on
    """
    pools: dict[str, list[dict]] = {'filtered': [], 'never_queued': [], 'mid_rated': []}
    for record in _listing_records.values():
        if record['check_status'] in ('too_old', 'already_applied', 'auth_required'):
            pools['filtered'].append(record)
        elif record['check_status'] == 'new' and not record['queued']:
            pools['never_queued'].append(record)
        elif record['rating'] in (2, 3):
            pools['mid_rated'].append(record)
    return pools


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
    def _result(status: str) -> dict:
        _check_status_counts[status] = _check_status_counts.get(status, 0) + 1
        _listing_records[(site, job_id)] = {
            'site': site, 'job_id': job_id, 'company': company, 'title': description,
            'date_posted': date_posted, 'check_status': status,
            'url': url or job_url(site, job_id),
            'queued': False, 'outcome': status, 'rating': None, 'summary': '',
        }
        logger.info(f'check_and_record_job: {status} — {company} — {description} '
                    f'[{site}/{job_id}, date_posted={date_posted!r}]')
        return {"content": [{"type": "text", "text": status}]}

    key = (site, job_id)
    if key in _processed_jobs:
        return _result("already_processed")

    matched_pdf = await company_matches_applied(company)
    if matched_pdf is not None:
        console.print(f"[dim]Skipping {company} — already applied ({matched_pdf}).[/dim]")
        return _result("already_applied")

    posted = parse_posting_date(date_posted)
    if posted and (date.today() - posted).days > JOB_MAX_AGE_DAYS:
        return _result("too_old")

    combined_text = ' '.join(filter(None, [description, content]))
    if _requires_current_us_auth(combined_text):
        console.print(f'[dim]Skipping {company} — requires current US work authorization.[/dim]')
        return _result("auth_required")

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
    return _result("new")


async def do_submit_job_extract(
    title: str, company: str, description: str,
    location: str | None = None, date_posted: str | None = None,
    closed: bool = False, salary: str | None = None, sponsorship_note: str | None = None,
    language_requirement: str | None = None, relocation: str | None = None,
    workplace_type: str | None = None, education_requirement: str | None = None,
) -> dict:
    _job_extracts.append({
        'title': title, 'company': company, 'description': description,
        'location': location or '', 'date_posted': date_posted or '',
        'closed': closed, 'salary': salary or '', 'sponsorship_note': sponsorship_note or '',
        'language_requirement': language_requirement or '', 'relocation': relocation or '',
        'workplace_type': (workplace_type or '').strip().lower(),
        'education_requirement': (education_requirement or '').strip().lower(),
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
    else:
        logger.warning(f'queue_candidate called without a query for {company} — {title}; '
                       'per-query counts will be understated')
    record = _listing_records.get((site, job_id))
    if record is not None:
        record['queued'] = True
        record['query'] = query
        record['url'] = url or record['url']
        record['snippet'] = snippet
        record['outcome'] = 'queued'
    logger.info(f'queue_candidate: {company} — {title} [{site}/{job_id}, query={query!r}]')
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
    "specific place (e.g. 'must be based in Portugal'); omit for work-from-anywhere roles. "
    "Pass workplace_type as exactly 'remote', 'hybrid', or 'onsite' when the page states the work "
    "arrangement; omit only if the page genuinely does not say. Any mention of required days in "
    "the office (e.g. '2-3 days onsite') is 'hybrid', not 'remote'. "
    "Pass education_requirement as 'master' or 'phd' ONLY if the posting states an advanced degree "
    "as a hard requirement (e.g. 'MSc in Computer Science required', 'PhD is a must'); omit it when "
    "the degree is merely preferred, when equivalent experience is accepted (\"Master's or equivalent "
    "practical experience\", 'MSc a plus', \"Bachelor's or Master's\"), or when only a Bachelor's is required.",
    {
        'type': 'object',
        'properties': {
            'title': {'type': 'string'},
            'company': {'type': 'string'},
            'description': {'type': 'string', 'description': 'Condensed posting content (requirements, responsibilities, stack, seniority)'},
            'location': {'type': 'string', 'description': 'Location including workplace type as shown, e.g. "Bucharest, Romania (Remote within country)"'},
            'workplace_type': {'type': 'string', 'description': "Work arrangement: 'remote', 'hybrid', or 'onsite'"},
            'date_posted': {'type': 'string'},
            'closed': {'type': 'boolean'},
            'salary': {'type': 'string'},
            'sponsorship_note': {'type': 'string'},
            'language_requirement': {'type': 'string', 'description': "Explicitly required languages, comma-separated lowercase, e.g. 'english, german'"},
            'relocation': {'type': 'string', 'description': 'Location the candidate must relocate to / reside in, if the posting requires one'},
            'education_requirement': {'type': 'string', 'description': "'master' or 'phd' if an advanced degree is a HARD requirement; empty when merely preferred or when equivalent experience is accepted"},
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
        workplace_type=args.get("workplace_type"),
        education_requirement=args.get("education_requirement"),
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
