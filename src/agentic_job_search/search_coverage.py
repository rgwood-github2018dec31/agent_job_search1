'''Measure Search Coverage: LinkedIn's own job emails against what the pipeline listed.

Three steps (see Measure Search Coverage in docs/requirements.md):

1. Harvest Alert Emails: a headless `claude -p` session holding only the Gmail read tools searches
   for LinkedIn's job emails and fetches each thread. The model only fetches. Code reads the raw
   tool results out of the session's stream and parses the job links itself, so no job id, title or
   date comes from a model's word. (Citation-exempt: the call predicts nothing about its input.)
2. Classify each emailed job against the Processed Job Records, the saved postings' ratings and the
   audit logs: seen and rated, seen but not rated, seen under a sibling id, outside the searched
   regions, or never seen.
3. Replay the fresh never-seen jobs in a searched region through `main.py --job-ids` and report any
   the rater scores COVERAGE_NOTIFY_MIN_RATING or more: jobs the User should have been sent.
'''

import json
import logging
import re
import subprocess
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from agentic_job_search import location as location_module
from agentic_job_search.config import (
    COVERAGE_ALERT_SENDERS,
    COVERAGE_GMAIL_SEARCH_PAGE_SIZE,
    COVERAGE_HARVEST_ATTEMPTS,
    COVERAGE_HARVEST_MAX_BUDGET_USD,
    COVERAGE_HARVEST_RETRY_WAIT_SECONDS,
    COVERAGE_HARVEST_TIMEOUT_SECONDS,
    COVERAGE_NOTIFY_MIN_RATING,
    MODEL_NAME_COVERAGE_HARVEST,
    REPLAY_SUMMARY_FILENAME,
)
from agentic_job_search.tools_generic import PROCESSED_JOBS_DIR, RUN_DIR, job_url

logger = logging.getLogger(__name__)

SEARCH_COVERAGE_DIR = RUN_DIR / 'search_coverage'
HARVESTED_THREADS_PATH = SEARCH_COVERAGE_DIR / 'harvested_threads.yaml'   # thread id -> email date
REPLAYED_JOBS_PATH = SEARCH_COVERAGE_DIR / 'replayed_jobs.yaml'           # job id -> replay outcome
LAST_COMPLETE_PATH = SEARCH_COVERAGE_DIR / 'last_complete.txt'            # ISO date of the last full check
ALERT_JOBS_FILENAME_PREFIX = 'linkedin_alerts-'
ALERT_JOBS_HEADER = ('job_id', 'title', 'company', 'location', 'email_date', 'email_kind', 'thread_id', 'subject')

GMAIL_TOOL_SEARCH_THREADS = 'mcp__claude_ai_Gmail__search_threads'
GMAIL_TOOL_GET_THREAD = 'mcp__claude_ai_Gmail__get_thread'
HARVEST_SETTINGS_JSON = json.dumps({'disableAllHooks': True})

EMAIL_KIND_APPLIED = 'applied'          # the job an application-confirmation email is about
EMAIL_KIND_RECOMMENDED = 'recommended'  # every other job LinkedIn put in an email

BUCKET_SEEN_RATED = 'seen_rated'
BUCKET_SEEN_NOT_RATED = 'seen_not_rated'
BUCKET_SIBLING_SEEN = 'sibling_seen'
BUCKET_OUT_OF_SEARCH_REGION = 'out_of_search_region'
BUCKET_NEVER_SEEN = 'never_seen'
BUCKETS = (BUCKET_SEEN_RATED, BUCKET_SEEN_NOT_RATED, BUCKET_SIBLING_SEEN,
           BUCKET_OUT_OF_SEARCH_REGION, BUCKET_NEVER_SEEN)
SEEN_BUCKETS = frozenset({BUCKET_SEEN_RATED, BUCKET_SEEN_NOT_RATED, BUCKET_SIBLING_SEEN})
REGION_NAME_EUROPEAN_UNION = 'european union'

# A job block in the plain-text body ends at its "View job:" line. `comm/` appears in email links.
VIEW_JOB_RE = re.compile(r'^\s*View job:\s*https://www\.linkedin\.com/(?:comm/)?jobs/view/(\d+)', re.MULTILINE)
APPLIED_SUBJECT_RE = re.compile(r'\byour application (?:was sent to|to)\b', re.IGNORECASE)
URL_LINE_RE = re.compile(r'^https?://')
# Title, company and location are the first three lines of the job's paragraph.
JOB_BLOCK_MIN_LINES = 3
SAVED_POSTING_RE = re.compile(r'^job_posting-(\d+)-rating_(\d+)-')
AUDIT_ROW_RE = re.compile(r'^\| (?P<company>[^|]*) \| (?P<title>[^|]*) \| (?P<status>[a-z_]+) \|.*/jobs/view/(?P<job_id>\d+)/')
AUDIT_QUERY_SECTION = '## 2. Search queries'
AUDIT_QUERY_LINE_RE = re.compile(r'^- `(?P<query>[^`]+)`')
# Claude Code saves a tool result too large for its context to a file and hands the model this path.
PERSISTED_RESULT_PATH_RE = re.compile(r'(/\S+/tool-results/\S+\.txt)')
ERROR_TEXT_HEAD_TAIL_CHARS = 2000
HEAD_AND_TAIL_PARTS = 2
PERCENT = 100
STDERR_TAIL_CHARS = 4000


@dataclass(frozen=True)
class AlertJob:
    '''One job LinkedIn put in an email, as the email's own text gives it.'''
    job_id: str
    title: str
    company: str
    location: str
    email_date: str     # ISO date the email was received
    email_kind: str     # EMAIL_KIND_APPLIED or EMAIL_KIND_RECOMMENDED
    thread_id: str
    subject: str


@dataclass
class HarvestResult:
    '''What one harvest session fetched, read from its stream by code.'''
    threads: dict[str, dict] = field(default_factory=dict)   # thread id -> get_thread JSON
    listed_thread_ids: set[str] = field(default_factory=set)  # every thread a search returned
    cost_usd: float = 0.0
    problems: list[str] = field(default_factory=list)
    gmail_tools_loaded: bool = False   # the session's init event listed the Gmail tools
    searched: bool = False             # at least one search_threads call returned


@dataclass
class ClassifiedJob:
    job: AlertJob
    bucket: str
    rating: int | None = None
    detail: str = ''     # audit status, sibling id, or the countries that put it out of region


def fold(text: str) -> str:
    '''Comparison key for company and title: case, punctuation and spacing folded.'''
    return ' '.join(re.sub(r'[^\w]+', ' ', str(text or '')).casefold().split())


def head_and_tail(text: str, chars: int = ERROR_TEXT_HEAD_TAIL_CHARS) -> str:
    '''Text for an error message; a long one keeps its head and tail and says what was cut.'''
    if len(text) <= HEAD_AND_TAIL_PARTS * chars:
        return text
    return f'{text[:chars]}\n[... {len(text) - HEAD_AND_TAIL_PARTS * chars} of {len(text)} chars cut ...]\n{text[-chars:]}'


# ---------------------------------------------------------------- parse the emails


def _job_labels(block: str) -> tuple[str, str, str]:
    '''(title, company, location) from the text before one "View job:" line, or blanks.'''
    paragraphs: list[list[str]] = [[]]
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            if paragraphs[-1]:
                paragraphs.append([])
            continue
        paragraphs[-1].append(stripped)
    for paragraph in reversed(paragraphs):
        lines = [line for line in paragraph if not URL_LINE_RE.match(line)]
        if len(lines) >= JOB_BLOCK_MIN_LINES and len(lines) == len(paragraph):
            return lines[0], lines[1], lines[2]
    return '', '', ''


def parse_alert_message(message: dict, thread_id: str) -> list[AlertJob]:
    '''Every job linked from one email message, in the order the email lists them.

    In an application-confirmation email the first job is the one applied to; the rest are
    LinkedIn's suggestions. A job whose labels cannot be read keeps its id with blank labels: the
    id is what the check needs.
    '''
    body = message.get('plaintextBody')
    if not isinstance(body, str):
        raise ValueError(f'Gmail message {message.get("id")!r} in thread {thread_id} has no plaintextBody '
                         f'(keys {sorted(message)})')
    subject = str(message.get('subject') or '')
    raw_date = str(message.get('date') or '')
    try:
        email_date = datetime.fromisoformat(raw_date.replace('Z', '+00:00')).date().isoformat()
    except ValueError as ex:
        raise ValueError(f'Gmail message {message.get("id")!r} in thread {thread_id} has an unreadable '
                         f'date {raw_date!r}: {ex}') from ex
    applied_email = bool(APPLIED_SUBJECT_RE.search(subject))
    jobs: list[AlertJob] = []
    previous_end = 0
    for index, match in enumerate(VIEW_JOB_RE.finditer(body)):
        title, company, location = _job_labels(body[previous_end:match.start()])
        previous_end = match.end()
        kind = EMAIL_KIND_APPLIED if applied_email and index == 0 else EMAIL_KIND_RECOMMENDED
        jobs.append(AlertJob(match.group(1), title, company, location, email_date, kind, thread_id, subject))
    return jobs


def parse_thread(thread: dict) -> list[AlertJob]:
    thread_id = str(thread.get('id') or '')
    messages = thread.get('messages')
    if not isinstance(messages, list):
        raise ValueError(f'Gmail thread {thread_id!r} has no messages list: {head_and_tail(json.dumps(thread))}')
    jobs: list[AlertJob] = []
    for message in messages:
        jobs.extend(parse_alert_message(message, thread_id))
    return jobs


def merge_alert_jobs(jobs: Iterable[AlertJob]) -> dict[str, AlertJob]:
    '''One entry per job id: the earliest email, except that an applied email always wins.'''
    merged: dict[str, AlertJob] = {}
    for job in jobs:
        current = merged.get(job.job_id)
        if current is None:
            merged[job.job_id] = job
            continue
        upgrades_to_applied = job.email_kind == EMAIL_KIND_APPLIED and current.email_kind != EMAIL_KIND_APPLIED
        earlier_of_same_kind = job.email_kind == current.email_kind and job.email_date < current.email_date
        if upgrades_to_applied or earlier_of_same_kind:
            merged[job.job_id] = job
    return merged


# ---------------------------------------------------------------- harvest through claude -p


def gmail_query(after: date) -> str:
    senders = ' OR '.join(COVERAGE_ALERT_SENDERS)
    return f'from:({senders}) after:{after:%Y/%m/%d}'


def build_harvest_prompt(after: date, skip_thread_ids: Iterable[str]) -> str:
    skip = sorted(set(skip_thread_ids))
    skip_text = ', '.join(skip) if skip else '(none)'
    return (
        f'Fetch email threads from Gmail. Do not summarise or interpret them.\n'
        f'1. Call {GMAIL_TOOL_SEARCH_THREADS} with query "{gmail_query(after)}", pageSize '
        f'{COVERAGE_GMAIL_SEARCH_PAGE_SIZE} and view THREAD_VIEW_IDS_ONLY. While the reply has a '
        f'nextPageToken, call it again with that pageToken.\n'
        f'2. For every thread id returned that is NOT in this skip list, call {GMAIL_TOOL_GET_THREAD} '
        f'with that threadId and messageFormat PLAIN_TEXT. Skip list: {skip_text}\n'
        f'3. When every such thread has been fetched, reply with the single word done.'
    )


def harvest_command(claude_executable: str, prompt: str) -> list[str]:
    return [
        claude_executable, '-p', prompt,
        '--model', MODEL_NAME_COVERAGE_HARVEST,
        '--output-format', 'stream-json', '--verbose',
        '--tools', '',                                    # no built-in tools: no Bash, no file access
        '--allowedTools', GMAIL_TOOL_SEARCH_THREADS, GMAIL_TOOL_GET_THREAD,
        '--permission-mode', 'dontAsk',                   # every other tool is denied, not asked
        # The claude.ai connectors load only from the user setting source ('' and 'local' load none, 2026-11-25).
        # Its hooks are switched off; its allow rules can add other MCP tools, which the stream check reports.
        '--setting-sources', 'user',
        '--settings', HARVEST_SETTINGS_JSON,
        '--no-session-persistence',
        '--max-budget-usd', str(COVERAGE_HARVEST_MAX_BUDGET_USD),
    ]


def _tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join(str(part.get('text', '')) for part in content if isinstance(part, dict))
    return json.dumps(content)


def _load_tool_json(text: str, tool_name: str, tool_input: dict) -> dict:
    '''The JSON a Gmail tool returned, also when Claude Code saved a large one to a file.'''
    try:
        return json.loads(text)
    except json.JSONDecodeError as inline_error:
        if (match := PERSISTED_RESULT_PATH_RE.search(text)) and Path(match.group(1)).is_file():
            saved_path = Path(match.group(1))
            saved_text = saved_path.read_text(encoding='utf-8')
            try:
                return json.loads(saved_text)
            except json.JSONDecodeError as ex:
                raise ValueError(f'{tool_name} {tool_input}: saved result {saved_path} is not JSON ({ex}): '
                                 f'{head_and_tail(saved_text)}') from ex
        raise ValueError(f'{tool_name} {tool_input}: result is not JSON ({inline_error}): '
                         f'{head_and_tail(text)}') from inline_error


def parse_harvest_stream(lines: Iterable[str]) -> HarvestResult:
    '''Read a `claude -p --output-format stream-json` transcript into the threads it fetched.'''
    result = HarvestResult()
    tool_uses: dict[str, tuple[str, dict]] = {}
    saw_result_event = False
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as ex:
            result.problems.append(f'stream line {line_number} is not JSON ({ex}): {head_and_tail(line)}')
            continue
        event_type = event.get('type')
        if event_type == 'system' and event.get('subtype') == 'init':
            tools = event.get('tools') or []
            result.gmail_tools_loaded = GMAIL_TOOL_SEARCH_THREADS in tools and GMAIL_TOOL_GET_THREAD in tools
        elif event_type == 'assistant':
            for part in event.get('message', {}).get('content', []):
                if part.get('type') == 'tool_use':
                    tool_uses[part.get('id', '')] = (part.get('name', ''), part.get('input') or {})
        elif event_type == 'user':
            for part in event.get('message', {}).get('content', []):
                if not isinstance(part, dict) or part.get('type') != 'tool_result':
                    continue
                tool_name, tool_input = tool_uses.get(part.get('tool_use_id', ''), ('(unknown tool)', {}))
                text = _tool_result_text(part.get('content'))
                if part.get('is_error'):
                    result.problems.append(f'{tool_name} {tool_input} returned an error: {head_and_tail(text)}')
                    continue
                if tool_name not in (GMAIL_TOOL_SEARCH_THREADS, GMAIL_TOOL_GET_THREAD):
                    result.problems.append(f'unexpected tool {tool_name} {tool_input}: {head_and_tail(text)}')
                    continue
                try:
                    payload = _load_tool_json(text, tool_name, tool_input)
                except ValueError as ex:
                    result.problems.append(str(ex))
                    continue
                if tool_name == GMAIL_TOOL_SEARCH_THREADS:
                    result.searched = True
                    threads = payload.get('threads', [])
                    if not isinstance(threads, list):
                        result.problems.append(f'{tool_name} {tool_input}: "threads" is not a list: '
                                               f'{head_and_tail(text)}')
                        continue
                    result.listed_thread_ids.update(str(thread.get('id')) for thread in threads if thread.get('id'))
                elif isinstance(payload.get('id'), str):
                    result.threads[payload['id']] = payload
                else:
                    result.problems.append(f'{tool_name} {tool_input}: no thread "id" in the result: '
                                           f'{head_and_tail(text)}')
        elif event_type == 'result':
            saw_result_event = True
            result.cost_usd = float(event.get('total_cost_usd') or 0.0)
            if event.get('subtype') != 'success' or event.get('is_error'):
                result.problems.append(f'harvest session ended with {event.get("subtype")!r}: '
                                       f'{head_and_tail(json.dumps(event))}')
    if not saw_result_event:
        result.problems.append('harvest stream has no result event: the session did not finish')
    return result


def _run_harvest_session(claude_executable: str, command: list[str], attempt: int) -> HarvestResult:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, cwd=SEARCH_COVERAGE_DIR,
                                   timeout=COVERAGE_HARVEST_TIMEOUT_SECONDS, check=False)
    except subprocess.TimeoutExpired as ex:
        raise RuntimeError(f'Harvest session {claude_executable} -p timed out after '
                           f'COVERAGE_HARVEST_TIMEOUT_SECONDS ({COVERAGE_HARVEST_TIMEOUT_SECONDS} s); stderr: '
                           f'{str(ex.stderr or "")[-STDERR_TAIL_CHARS:]}') from ex
    except OSError as ex:
        raise RuntimeError(f'Harvest session could not start {claude_executable}: {type(ex).__name__}: {ex}') from ex
    stream_path = SEARCH_COVERAGE_DIR / f'harvest-{datetime.now():%Y%m%d-%H%M%S}-attempt{attempt}.jsonl'
    stream_path.write_text(completed.stdout, encoding='utf-8')
    result = parse_harvest_stream(completed.stdout.splitlines())
    if completed.returncode != 0:
        raise RuntimeError(f'Harvest session exited {completed.returncode}; stream saved to {stream_path}; '
                           f'problems {result.problems}; stderr tail: {completed.stderr[-STDERR_TAIL_CHARS:]}')
    logger.info(f'Harvest Alert Emails: attempt {attempt}: Gmail tools loaded {result.gmail_tools_loaded}, '
                f'searched {result.searched}, {len(result.threads)} thread(s) fetched, '
                f'{len(result.listed_thread_ids)} listed, ${result.cost_usd:.4f}, stream {stream_path}')
    return result


def run_harvest(claude_executable: str, after: date, skip_thread_ids: set[str]) -> HarvestResult:
    '''Fetch every LinkedIn job email since `after` not already harvested. Raises if no session could search.'''
    prompt = build_harvest_prompt(after, skip_thread_ids)
    command = harvest_command(claude_executable, prompt)
    logger.info(f'Harvest Alert Emails: {MODEL_NAME_COVERAGE_HARVEST} via {claude_executable} -p, query '
                f'"{gmail_query(after)}", {len(skip_thread_ids)} thread(s) already harvested')
    SEARCH_COVERAGE_DIR.mkdir(parents=True, exist_ok=True)
    total_cost = 0.0
    for attempt in range(1, COVERAGE_HARVEST_ATTEMPTS + 1):
        result = _run_harvest_session(claude_executable, command, attempt)
        total_cost += result.cost_usd
        if result.gmail_tools_loaded and result.searched:
            break
        reason = ('the Gmail connector tools were not in the session' if not result.gmail_tools_loaded
                  else 'the session never ran a search')
        if attempt == COVERAGE_HARVEST_ATTEMPTS:
            raise RuntimeError(f'Harvest Alert Emails failed: {reason} in all COVERAGE_HARVEST_ATTEMPTS '
                               f'({COVERAGE_HARVEST_ATTEMPTS}) sessions; last problems {result.problems}. '
                               f'Check `claude mcp list` shows "claude.ai Gmail" connected')
        logger.warning(f'Harvest Alert Emails: attempt {attempt}: {reason}; starting again in '
                       f'COVERAGE_HARVEST_RETRY_WAIT_SECONDS ({COVERAGE_HARVEST_RETRY_WAIT_SECONDS} s)')
        time.sleep(COVERAGE_HARVEST_RETRY_WAIT_SECONDS)
    result.cost_usd = total_cost
    # Completeness is checked in code: every thread a search listed and not skipped must be fetched.
    unread = sorted(result.listed_thread_ids - skip_thread_ids - set(result.threads))
    if unread:
        result.problems.append(f'{len(unread)} listed thread(s) were never fetched: {unread}')
    return result


# ---------------------------------------------------------------- ledgers and alert files


def load_yaml_mapping(path: Path) -> dict:
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding='utf-8'))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f'{path} should hold a mapping, found {type(loaded).__name__}')
    return loaded


def save_yaml_mapping(path: Path, mapping: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(mapping, allow_unicode=True, sort_keys=True), encoding='utf-8')


def write_alert_jobs(path: Path, jobs: Iterable[AlertJob]) -> None:
    rows = ['\t'.join(ALERT_JOBS_HEADER)]
    for job in jobs:
        values = asdict(job)
        rows.append('\t'.join(' '.join(str(values[column]).split()) for column in ALERT_JOBS_HEADER))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(rows) + '\n', encoding='utf-8')


def read_alert_jobs(directory: Path) -> list[AlertJob]:
    jobs: list[AlertJob] = []
    for path in sorted(directory.glob(f'{ALERT_JOBS_FILENAME_PREFIX}*.tsv')):
        lines = path.read_text(encoding='utf-8').splitlines()
        if not lines or tuple(lines[0].split('\t')) != ALERT_JOBS_HEADER:
            raise ValueError(f'{path}: expected the header {ALERT_JOBS_HEADER}, found {lines[:1]}')
        for line_number, line in enumerate(lines[1:], start=2):
            fields = line.split('\t')
            if len(fields) != len(ALERT_JOBS_HEADER) or not fields[0].isdigit():
                raise ValueError(f'{path}:{line_number}: malformed row {line!r}')
            jobs.append(AlertJob(*fields))
    return jobs


# ---------------------------------------------------------------- what the pipeline saw


@dataclass
class PipelineRecord:
    '''What the pipeline knows about the jobs it listed.'''
    processed_ids: set[str] = field(default_factory=set)
    ids_by_company_title: dict[tuple[str, str], str] = field(default_factory=dict)
    ratings: dict[str, int] = field(default_factory=dict)
    audit_statuses: dict[str, str] = field(default_factory=dict)


def load_pipeline_record(run_dir: Path = RUN_DIR, processed_dir: Path = PROCESSED_JOBS_DIR) -> PipelineRecord:
    record = PipelineRecord()
    for path in processed_dir.glob('*.yaml'):
        try:
            data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        except yaml.YAMLError as ex:
            raise ValueError(f'Processed Job Record {path} is not YAML: {ex}') from ex
        job_id = str(data.get('job_id') or '')
        if not job_id:
            raise ValueError(f'Processed Job Record {path} has no job_id: {data}')
        record.processed_ids.add(job_id)
        key = (fold(data.get('company', '')), fold(data.get('description', '')))
        if all(key):
            record.ids_by_company_title.setdefault(key, job_id)
    for saved_dir in run_dir.glob('saved_jobs-*'):
        for path in saved_dir.glob('job_posting-*.md'):
            if match := SAVED_POSTING_RE.match(path.name):
                record.ratings[match.group(1)] = int(match.group(2))
    for audit_path in sorted((run_dir / 'audit_logs').glob('audit-*.md'), key=lambda path: path.stat().st_mtime):
        for line in audit_path.read_text(encoding='utf-8', errors='replace').splitlines():
            if match := AUDIT_ROW_RE.match(line):
                record.audit_statuses[match.group('job_id')] = match.group('status')
    return record


def latest_search_queries(run_dir: Path = RUN_DIR) -> list[str]:
    '''The Search Queries of the newest run audit log (empty when there is none).'''
    audit_paths = sorted((run_dir / 'audit_logs').glob('audit-*.md'), key=lambda path: path.stat().st_mtime)
    if not audit_paths:
        return []
    queries: list[str] = []
    in_section = False
    for line in audit_paths[-1].read_text(encoding='utf-8', errors='replace').splitlines():
        if line.startswith('## '):
            in_section = line.startswith(AUDIT_QUERY_SECTION)
            continue
        if in_section and (match := AUDIT_QUERY_LINE_RE.match(line)):
            queries.append(match.group('query'))
    return queries


def in_search_regions(classification: dict, region_names: Iterable[str]) -> bool:
    '''True unless the classifier named countries and none of them is in a searched region.

    An unknown or multi-country area counts as in region: a coverage check that fails permissive
    replays a job too many, never hides a miss.
    '''
    countries = classification.get('countries') or []
    if not countries or classification.get('broad_area'):
        return True
    for region in (fold(name) for name in region_names):
        for country in countries:
            if region == REGION_NAME_EUROPEAN_UNION and location_module.is_eu_member(country):
                return True
            if fold(country) == region:
                return True
    return False


async def classify_jobs(
    jobs: Iterable[AlertJob], record: PipelineRecord, region_names: list[str],
    classify_location: Callable[[str], Awaitable[dict]] = location_module.classify_location,
) -> list[ClassifiedJob]:
    classified: list[ClassifiedJob] = []
    for job in jobs:
        if job.job_id in record.processed_ids:
            rating = record.ratings.get(job.job_id)
            bucket = BUCKET_SEEN_RATED if rating is not None else BUCKET_SEEN_NOT_RATED
            classified.append(ClassifiedJob(job, bucket, rating, record.audit_statuses.get(job.job_id, '')))
            continue
        sibling = record.ids_by_company_title.get((fold(job.company), fold(job.title)))
        if sibling:
            classified.append(ClassifiedJob(job, BUCKET_SIBLING_SEEN, record.ratings.get(sibling),
                                            f'sibling {sibling}'))
            continue
        if region_names and job.location:
            geography = await classify_location(job.location)
            if not in_search_regions(geography, region_names):
                classified.append(ClassifiedJob(job, BUCKET_OUT_OF_SEARCH_REGION, None,
                                                ', '.join(geography.get('countries') or [])))
                continue
        classified.append(ClassifiedJob(job, BUCKET_NEVER_SEEN))
    return classified


# ---------------------------------------------------------------- replay


def choose_replay_jobs(classified: Iterable[ClassifiedJob], already_replayed: set[str], cap: int) -> tuple[list[AlertJob], int]:
    '''Never-seen jobs not replayed before: applied first, then newest. Returns (chosen, left out).'''
    pending = [item.job for item in classified
               if item.bucket == BUCKET_NEVER_SEEN and item.job.job_id not in already_replayed]
    applied = [job for job in pending if job.email_kind == EMAIL_KIND_APPLIED]
    recommended = sorted((job for job in pending if job.email_kind != EMAIL_KIND_APPLIED),
                         key=lambda job: job.email_date, reverse=True)
    ordered = applied + recommended
    return ordered[:cap], max(len(ordered) - cap, 0)


def write_job_ids_file(path: Path, jobs: Iterable[AlertJob]) -> None:
    lines = ['# Measure Search Coverage: never-seen jobs from LinkedIn emails (job_id, title, company)']
    lines += ['\t'.join([job.job_id, ' '.join(job.title.split()), ' '.join(job.company.split())]) for job in jobs]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def read_replay_summary(replay_dir: Path) -> dict[str, dict]:
    '''job id -> {outcome, rating, summary} from a --job-ids replay's summary.tsv.'''
    path = replay_dir / REPLAY_SUMMARY_FILENAME
    rows: dict[str, dict] = {}
    lines = path.read_text(encoding='utf-8').splitlines()
    header = lines[0].split('\t') if lines else []
    for line_number, line in enumerate(lines[1:], start=2):
        fields = dict(zip(header, line.split('\t'), strict=False))  # a blank last cell may be dropped
        if not fields.get('job_id', '').isdigit():
            raise ValueError(f'{path}:{line_number}: no job id in {line!r}')
        rating_text = fields.get('rating', '')
        rows[fields['job_id']] = {
            'outcome': fields.get('outcome', ''),
            'rating': int(rating_text) if rating_text.isdigit() else None,
            'summary': fields.get('summary', ''),
        }
    return rows


# ---------------------------------------------------------------- report


def coverage_ratio(classified: Iterable[ClassifiedJob]) -> tuple[int, int]:
    '''(seen, seen + never seen) over the jobs in a searched region.'''
    counts = Counter(item.bucket for item in classified)
    seen = sum(counts[bucket] for bucket in SEEN_BUCKETS)
    return seen, seen + counts[BUCKET_NEVER_SEEN]


def _row(item: ClassifiedJob, replayed: dict[str, dict]) -> str:
    job = item.job
    replay = replayed.get(job.job_id, {})
    rating = item.rating if item.rating is not None else replay.get('rating')
    outcome = replay.get('outcome', '') if job.job_id in replayed else item.detail
    cells = [job.email_date, job.email_kind, job.title or '?', job.company or '?', job.location or '?',
             '' if rating is None else str(rating), outcome, job_url('linkedin', job.job_id)]
    return '| ' + ' | '.join(cell.replace('|', '/') for cell in cells) + ' |'


def build_report(classified: list[ClassifiedJob], replayed: dict[str, dict], queries: list[str],
                 window: tuple[date, date], notes: list[str]) -> str:
    counts = Counter(item.bucket for item in classified)
    seen, in_region = coverage_ratio(classified)
    percent = f'{PERCENT * seen / in_region:.0f}%' if in_region else 'n/a'
    lines = [
        f'# Search coverage — LinkedIn emails {window[0]} .. {window[1]}',
        '',
        f'**{seen} of {in_region}** emailed jobs in a searched region were listed by the pipeline ({percent}).',
        '',
        '| bucket | jobs |', '|---|---|',
        *(f'| {bucket} | {counts[bucket]} |' for bucket in BUCKETS),
        '',
    ]
    if notes:
        lines += ['## Notes', '', *(f'- {note}' for note in notes), '']
    should_have_had = [item for item in classified
                       if (replayed.get(item.job.job_id, {}).get('rating') or 0) >= COVERAGE_NOTIFY_MIN_RATING]
    lines += [f'## Missed jobs rated {COVERAGE_NOTIFY_MIN_RATING}+ on replay', '']
    lines += [_row(item, replayed) for item in should_have_had] or ['None.']
    lines += ['', '## Never seen', '',
              '| emailed | kind | title | company | location | rating | outcome | url |', '|---|---|---|---|---|---|---|---|']
    lines += [_row(item, replayed) for item in classified if item.bucket == BUCKET_NEVER_SEEN]
    never_seen = [item.job for item in classified if item.bucket == BUCKET_NEVER_SEEN]
    repeated_titles = [(title, n) for title, n in Counter(fold(job.title) for job in never_seen if job.title).most_common() if n > 1]
    repeated_companies = [(name, n) for name, n in Counter(job.company for job in never_seen if job.company).most_common() if n > 1]
    lines += ['', '## Missed more than once', '']
    lines += [f'- title "{title}": {n}' for title, n in repeated_titles]
    lines += [f'- company "{name}": {n}' for name, n in repeated_companies]
    if not repeated_titles and not repeated_companies:
        lines.append('None.')
    lines += ['', '## Search Queries of the latest run', '', *(f'- `{query}`' for query in queries), '',
              '## Seen', '', '| emailed | kind | title | company | location | rating | status | url |',
              '|---|---|---|---|---|---|---|---|']
    lines += [_row(item, replayed) for item in classified if item.bucket in SEEN_BUCKETS]
    lines += ['', '## Outside the searched regions', '',
              '| emailed | kind | title | company | location | rating | countries | url |', '|---|---|---|---|---|---|---|---|']
    lines += [_row(item, replayed) for item in classified if item.bucket == BUCKET_OUT_OF_SEARCH_REGION]
    return '\n'.join(lines) + '\n'


def build_telegram_summary(classified: list[ClassifiedJob], replayed_now: dict[str, dict],
                           report_path: Path, notes: list[str]) -> str:
    seen, in_region = coverage_ratio(classified)
    never_seen = sum(1 for item in classified if item.bucket == BUCKET_NEVER_SEEN)
    applied_missed = sum(1 for item in classified
                         if item.bucket == BUCKET_NEVER_SEEN and item.job.email_kind == EMAIL_KIND_APPLIED)
    lines = [f'🔎 Search coverage: {seen}/{in_region} emailed in-region jobs listed; {never_seen} never seen '
             f'({applied_missed} you applied to); {len(replayed_now)} replayed now.']
    by_id = {item.job.job_id: item.job for item in classified}
    for job_id, replay in replayed_now.items():
        if (replay.get('rating') or 0) >= COVERAGE_NOTIFY_MIN_RATING and job_id in by_id:
            job = by_id[job_id]
            lines.append(f'⚠️ Missed, rated {replay["rating"]}: {job.title} at {job.company} — {job_url("linkedin", job_id)}')
    lines += [f'Note: {note}' for note in notes]
    lines.append(f'Report: {report_path}')
    return '\n'.join(lines)


def window_for(today: date, window_days: int, settle_days: int) -> tuple[date, date]:
    '''(first, last) email date measured: old enough for a run to have listed it, new enough to replay.'''
    return today - timedelta(days=window_days), today - timedelta(days=settle_days)
