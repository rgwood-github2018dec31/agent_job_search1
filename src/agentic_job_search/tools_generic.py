
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import aclosing
from datetime import date, datetime, timedelta
from http import HTTPStatus
from pathlib import Path
from typing import Any

import pypdf
import yaml
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ResultMessage,
    create_sdk_mcp_server,
    tool,
)
from claude_agent_sdk import (
    query as sdk_query,
)
from rich.console import Console
from utils_tools_n_agents_common.mcp_client import unwrap_exception
from utils_tools_n_agents_common.models import ANTHROPIC_MODEL_NAME_LOW, route_for

import agentic_job_search.preferences as preferences
from agentic_job_search import location_review
from agentic_job_search.config import (
    APPLIED_JOBS_HORIZON_DAYS,
    BLACKLIST_CONTEXT_DESCRIPTION_MAX_CHARS,
    EVAL_ERROR_MAX_RELEASES,
    JOB_MAX_AGE_DAYS,
    MODEL_NAME_COMPANY_MATCH,
    PDF_PROMPT_MAX_CHARS,
    RAW_POSTING_MAX_CHARS,
    RAW_POSTINGS_RETENTION_DAYS,
    RECRUITER_REPOST_MAX_PRIORS,
    RECRUITER_REPOST_WINDOW_DAYS,
    SALARY_FIELD_DESCRIPTION,
    SCRAPER_DATE_POSTED_LABEL,
    SCRAPER_DATE_POSTED_SECONDS,
    SCRAPER_EXPERIENCE_LABEL,
    SCRAPER_MAX_LISTINGS_PER_SEARCH,
    SECONDS_PER_MINUTE,
    STATED_WORKING_LANGUAGE_FIELD_DESCRIPTION,
    UI_BLOCK_SIGNATURES,
    UI_CHIP_CHOSEN_VALUES,
    UI_CONTRACT_ELEMENTS,
    UI_FINGERPRINT_FILENAME,
    UI_STRUCTURAL_CHIPS,
    is_mid_rated,
)
from agentic_job_search.text_budget import pages_to_prompt, snippet, truncate_reported
from agentic_job_search.triage import chat_openrouter, extract_json_object

console = Console()

PROJECT_DIR = Path(__file__).parent.parent.parent  # src/agentic_job_search/ -> src/ -> project root
RUN_DIR = PROJECT_DIR / 'run_dir'
JOB_REQUIREMENTS_PATH = RUN_DIR / 'JOB_REQUIREMENTS.md'
PROCESSED_JOBS_DIR = RUN_DIR / 'processed_jobs'
# The page an extract was made from, kept so a figure it reports can be checked against its source.
# READ BY NOTHING IN THE PIPELINE — no gate, no rating, no dedup. It exists because 'CA$208,580 -
# $273,770' reached a notification with no way, anywhere, to tell whether the page said that
# (2026-09-22). Date-partitioned so pruning is a whole-directory operation.
RAW_POSTINGS_DIR = RUN_DIR / 'raw_postings'
# Applied-job PDFs travel between TWO DISTINCT DIRECTORIES, and nothing conflates them:
#
#   save_dir       — SOURCE. Outside the project, user-controlled, configurable via the
#                    `save_dir` preference (default ~/Downloads). Where the browser drops a
#                    saved job posting. The agent only ever globs it and MOVES files OUT.
#   APPLIED_JOBS_DIR — DESTINATION. Inside run_dir/, agent-owned, never configurable. The
#                    applied-jobs corpus that feeds query generation, the ideal-role profile,
#                    and the already-applied blocklist. Files here are date-prefixed and read.
#
# Functions that touch both take them as separate parameters named `save_dir` and
# `applied_to_dir`; a function that names only one directory operates only on that one.
APPLIED_JOBS_DIR = RUN_DIR / 'applied_jobs'
APPLIED_JOBS_INDEX_PATH = APPLIED_JOBS_DIR / 'index.yaml'
# Legacy per-path cache from when the corpus lived in ~/Downloads; still read during ingest
# as a fallback source of applied dates if a file's mtime has drifted.
LEGACY_DOWNLOADS_CACHE_PATH = RUN_DIR / 'downloads_pdf_cache.yaml'

# Leading wildcard so an already-date-prefixed PDF landing back in the save directory is ingested
# (with its original date) rather than silently ignored.
APPLIED_PDF_GLOB = '*cat-saved_jd-*.pdf'
_DATE_PREFIX_RE = re.compile(r'^(\d{4}-\d{2}-\d{2})-')

# index.yaml caches a PDF's extraction keyed by mtime; filesystems round mtimes differently, so
# a difference under this is the same file.
INDEX_MTIME_TOLERANCE_SECONDS = 1.0
# Saved-job filename components. A long description (e.g. a full triage-reason sentence) would
# otherwise push the filename past the filesystem's 255-byte limit; the file body keeps it all.
SAVED_JOB_FILENAME_COMPANY_MAX_CHARS = 40
SAVED_JOB_FILENAME_DESCRIPTION_MAX_CHARS = 120
JSON_INDENT = 2
# Region overlap needs at least two regions' result sets to compare.
MIN_REGIONS_FOR_OVERLAP = 2
REGION_OVERLAP_DECIMALS = 3

logger = logging.getLogger(__name__)

_processed_jobs: set[tuple[str, str]] = set()
_candidates: list[dict] = []
_candidates_per_query: dict[str, int] = {}
_applied_companies: dict[str, str] = {}  # company_name -> PDF filename
# Per-page extracted text of each in-horizon applied job, newest first, for the reference profile.
# Kept as pages until a prompt serializes it (text_budget.pages_to_prompt), so size is reportable.
_reference_job_pages: list[list[str]] = []
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
# query -> why it failed. Kept separate from the 'error' sentinel in _queries_searched, which the
# audit log renders as **FAILED** and which must stay a bare sentinel. Without this the CAUSE of a
# failed query reached the run log and nothing else: on 2026-09-02 six queries died on a 402 and
# neither the audit log nor the Telegram summary carried a single byte of the reason.
_query_errors: dict[str, str] = {}
# query -> {status: count}: per-query breakdown of check_and_record_job outcomes, so a
# dedup-saturated query is distinguishable from one that barely ran. The run-global
# _check_status_counts cannot make that distinction, which is why a run that inspected 7
# listings instead of 166 read as ordinary dedup.
_check_status_per_query: dict[str, dict[str, int]] = {}
# The base Search Query Stage 1b is currently running, set by run_scraper. Candidates are
# attributed to this rather than to the query string the model passes back, which is the full
# search text and so never matches.
_current_query: str | None = None

# --- Per-search coverage bookkeeping -----------------------------------------------------------
#
# `_check_status_counts` counts check_and_record_job CALLS, not distinct jobs. When two regions
# return the same 25 cards that reads as "50 listings inspected" — indistinguishable from genuine
# coverage of 50, and it pushes the query FURTHER from the low-yield retry threshold. So distinct
# ids are tracked separately, per run and per (query, region).
_distinct_listing_ids: set[str] = set()                      # run-global distinct (site, job_id) keys
_search_ids: dict[tuple[str, str], set[str]] = {}            # (query, region) -> harvested job ids
# One record per search: region, result count, contract verdict. Feeds the audit log and funnel.
_search_reports: list[dict] = []
# Contract/blocking problems found this run, surfaced in the audit log and the Telegram summary.
_ui_alerts: list[dict] = []
# Per-query Stage 1b cost/iterations, so a scraper regression is attributable to a query rather
# than visible only as a stage total.
_scrape_per_query: dict = {}
# The region whose search is currently being walked, set by report_search. Listings are attributed
# to (query, region) so region overlap is measurable; without it the run-global counts cannot tell
# "two regions, 25 jobs each" from "two regions, the SAME 25 jobs".
_current_region: str | None = None


# queue-time skip reason -> count; folded into the run funnel.
_queue_skipped_counts: dict[str, int] = {}

# Titles that are unambiguously below the target seniority. LinkedIn used to enforce this
# server-side via `f_E=4%2C5%2C6`, but it now ignores filter params in the search URL, so these
# reach the queue and would each cost a Stage 2a extract plus a rating call.
#
# Deliberately conservative — an auto-skip is unappealable, and the search card gives only a
# title. 'associate' and bare 'graduate' are NOT here: "Associate Director" and "Graduate
# Research Scientist" are senior in many orgs. Work arrangement is not filtered here at all;
# AGENTS.md records that workplace type must be resolved from the job page, not the card.
_JUNIOR_TITLE_RE = re.compile(
    r'\b(intern|interns|internship|junior|jr|entry[ -]level|new grad(uate)?|'
    r'apprentice|trainee|co[ -]op|working student)\b',
    re.IGNORECASE,
)


def format_status_counts(counts: dict[str, int]) -> str:
    """e.g. 'already_processed 19, new 8' — for the Stage 1b log line and the audit log."""
    return ', '.join(f'{status} {n}' for status, n in sorted(counts.items())) or 'none'


def title_rejection_reason(title: str) -> str | None:
    """Why this listing title should not enter the paid Stage 2 path, or None to proceed.

    A $0 check on text already captured from the search card. Both rules replace filtering
    LinkedIn no longer does: `f_E` for seniority, and the user's own excluded-title
    preference for people-management roles.
    """
    if not title:
        return None
    if (match := _JUNIOR_TITLE_RE.search(title)):
        return f'below target seniority ({match.group(0).lower()!r} in title)'
    for word in preferences.excluded_title_words():
        if re.search(rf'\b{re.escape(word)}\b', title, re.IGNORECASE):
            return f'excluded title word ({word!r})'
    return None


class AgentApiError(Exception):
    """An Anthropic API call failed at the HTTP level (401 / 429 / 500 / 529 ...).

    The SDK reports this as a ResultMessage with is_error=True and subtype=="success", putting the
    real status in api_error_status. Its own fallback text renders that as the self-contradictory
    "Claude Code returned an error result: success" — the word "success" is the subtype leaking
    through because the `errors` list was empty, not a status. Nine identical copies of that string
    are what this exception exists to replace.

    Unlike a corrupt PDF, this is systemic: it fails every call the same way, so callers abort
    rather than retrying per file.
    """

    def __init__(self, detail: str, api_error_status: int | None = None):
        super().__init__(detail)
        self.api_error_status = api_error_status


_AUTH_FAILURE_STATUSES = {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}
# 529 is Anthropic's "overloaded", which has no HTTPStatus member.
ANTHROPIC_OVERLOADED_STATUS = 529
_TRANSIENT_FAILURE_STATUSES = {
    HTTPStatus.TOO_MANY_REQUESTS, HTTPStatus.INTERNAL_SERVER_ERROR, HTTPStatus.BAD_GATEWAY,
    HTTPStatus.SERVICE_UNAVAILABLE, ANTHROPIC_OVERLOADED_STATUS,
}


def _raise_if_result_error(msg: ResultMessage, context: str) -> None:
    """Raise AgentApiError if the SDK reported an API-level failure for this result.

    The errored ResultMessage arrives BEFORE the SDK's trailing, uninformative ProcessError, so
    raising here is what replaces "error result: success" with the actual status code.
    """
    if not getattr(msg, 'is_error', False):
        return
    status = getattr(msg, 'api_error_status', None)
    parts = [f'API error {status}' if status else 'API error']
    parts.append(f'subtype={getattr(msg, "subtype", None)!r}')
    if (terminal_reason := getattr(msg, 'terminal_reason', None)):
        parts.append(f'terminal_reason={terminal_reason!r}')
    if (errors := getattr(msg, 'errors', None)):
        parts.append(f'errors={"; ".join(errors)}')
    if status in _AUTH_FAILURE_STATUSES:
        parts.append('not authenticated — run `claude /login`')
    elif status in _TRANSIENT_FAILURE_STATUSES:
        parts.append('rate limited or overloaded — retry later')
    elif status is None:
        # Seen 2026-09-21 on a lapsed login: api_error with no status, so neither hint above fired.
        parts.append('no HTTP status reported — check the Claude login (`claude /login`) and network')
    detail = f'{context}: ' + ' | '.join(parts)
    raise AgentApiError(detail, api_error_status=status)


async def _extract_applied_job_metadata(text: str, filename: str) -> dict:
    """Extract company, title, and recruiter-agency signal from an applied-job PDF's text.

    Recruiting agencies and job aggregators repost roles on behalf of other companies, so the
    name on the posting is often not the employer. Blocklisting such a name would suppress every
    future listing that agency posts, hence the explicit is_recruiting_agency / end_client_name.
    """
    options = ClaudeAgentOptions(
        model=ANTHROPIC_MODEL_NAME_LOW,
        tools=[],
        permission_mode='bypassPermissions',
        setting_sources=[],
        strict_mcp_config=True,
        skills=[],
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
        f'Job posting text:\n{text}\n\n'
        'Identify the organization posting this job, the job title, whether the poster is a '
        'recruiting agency/aggregator rather than the hiring employer, and the end client if named.'
    )
    async with aclosing(sdk_query(prompt=prompt, options=options)) as stream:
        async for msg in stream:
            if not isinstance(msg, ResultMessage):
                continue
            _raise_if_result_error(msg, f'Applied-job metadata extraction failed for {filename}')
            if msg.structured_output:
                out = msg.structured_output
                return {
                    'company': out.get('company_name', ''),
                    'job_title': out.get('job_title', ''),
                    'is_agency': bool(out.get('is_recruiting_agency', False)),
                    'end_client': out.get('end_client_name', ''),
                }
    logger.warning(f'No structured output extracting applied-job metadata for {filename}')
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
    """Fuzzy company match via the OpenRouter MCP server. Raises on any failure."""
    content, _cost = await chat_openrouter(
        f'Does "{candidate}" refer to the same organization as any of these companies?\n\n'
        f'{companies_list}\n\n'
        'Respond with ONLY a JSON object: '
        '{"matches": <true|false>, "matched_company_name": "<exact name from the list, or empty string>"}',
        model=MODEL_NAME_COMPANY_MATCH,
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
        logger.debug(f'company_matches_applied: exact normalized match {candidate!r} -> {exact!r} (no LLM call)')
        return _applied_companies[exact]

    companies_list = '\n'.join(f'- {name}' for name in _applied_companies)

    if route_for(MODEL_NAME_COMPANY_MATCH) == 'openrouter':
        try:
            structured = await _company_matches_openrouter(candidate, companies_list)
            if not structured.get('matches'):
                return None
            return _applied_companies.get(structured.get('matched_company_name', ''), '<unknown PDF>')
        except Exception as ex:
            logger.warning(f'company_matches_applied via OpenRouter failed, falling back to Anthropic: {ex}')

    options = ClaudeAgentOptions(
        model=ANTHROPIC_MODEL_NAME_LOW,
        tools=[],
        permission_mode='bypassPermissions',
        setting_sources=[],
        strict_mcp_config=True,
        skills=[],
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
    async with aclosing(sdk_query(
            prompt=f'Does "{candidate}" refer to the same organization as any of these companies?\n\n{companies_list}',
            options=options,
        )) as stream:
        async for msg in stream:
            if not isinstance(msg, ResultMessage):
                continue
            _raise_if_result_error(msg, f'Company match check failed for {candidate!r}')
            if msg.structured_output:
                structured = msg.structured_output
    if not structured or not structured.get('matches'):
        return None
    matched = structured.get('matched_company_name', '')
    return _applied_companies.get(matched, '<unknown PDF>')


def _blacklist_confirm_prompt(company: str, entry_name: str, reason: str, context: str) -> str:
    return (
        f'A job posting lists its company as "{company}".\n\n'
        f'The user has blacklisted an organization they refer to as "{entry_name}"'
        + (f' (their reason: {reason})' if reason else '')
        + '.\n\n'
        'Is the organization in this posting the SAME organization the user blacklisted, or a '
        'different one that happens to share the name?\n\n'
        f'Posting context:\n{context or "(none available)"}\n\n'
        'Respond with ONLY a JSON object: {"same_organization": <true|false>}'
    )


async def _blacklist_confirms_openrouter(prompt: str) -> dict:
    """Confirm a blacklist name hit via the OpenRouter MCP server. Raises on any failure."""
    content, _cost = await chat_openrouter(prompt, model=MODEL_NAME_COMPANY_MATCH)
    return extract_json_object(content)


async def company_blacklist_reason(company: str, context: str = '', description: str = '') -> str | None:
    """Return the blacklist reason if this posting is from a blacklisted company, else None.

    Deliberately does NOT reuse ``company_matches_applied``: that one matches on a normalized,
    lowercased name and fuzzy-matches by default, both of which are wrong for a standing
    user-authored blacklist. Here the trigger is a **case-sensitive exact** name match, which
    costs nothing and cannot over-block, followed by one cheap LLM call to confirm the posting is
    really that organization rather than a same-named one ("Cohere" the AI lab is not Cohere
    Health). No name hit means no LLM call at all, so the gate is free on almost every posting.

    Ambiguity resolves toward rejecting: if the confirmation call fails for any reason, the exact
    name match stands and the job is rejected. A blacklist is an explicit user decision, and an
    unrelated tool-server outage must not quietly let a blacklisted company back through.

    `description` is budgeted only after a name hit, because only then does anything read it.
    Budgeting it up front logged a truncation for every job on every run (20 on 2026-09-24) for
    text that was then thrown away unread.
    """
    entries = preferences.blacklisted_companies()
    if not entries or not company or not company.strip():
        return None

    candidate = company.strip()
    matched = next(((name, reason) for name, reason in entries if name == candidate), None)
    if matched is None:
        return None
    entry_name, reason = matched
    label = reason or 'blacklisted'

    if description:
        context = '\n'.join(filter(None, [context, truncate_reported(
            description, BLACKLIST_CONTEXT_DESCRIPTION_MAX_CHARS,
            f'blacklist-confirmation description for {candidate!r}',
        )]))
    prompt = _blacklist_confirm_prompt(candidate, entry_name, reason, context)
    structured: dict | None = None

    if route_for(MODEL_NAME_COMPANY_MATCH) == 'openrouter':
        try:
            structured = await _blacklist_confirms_openrouter(prompt)
        except Exception as ex:
            logger.warning(f'Blacklist confirmation via OpenRouter failed for {candidate!r}, falling back to Anthropic: {ex}')

    if structured is None:
        options = ClaudeAgentOptions(
            model=ANTHROPIC_MODEL_NAME_LOW,
            tools=[],
            permission_mode='bypassPermissions',
            setting_sources=[],
            strict_mcp_config=True,
            skills=[],
            output_format={
                'type': 'json_schema',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'same_organization': {
                            'type': 'boolean',
                            'description': 'True if the posting is from the blacklisted organization',
                        },
                    },
                    'required': ['same_organization'],
                },
            },
            cwd=str(PROJECT_DIR),
        )
        try:
            async with aclosing(sdk_query(prompt=prompt, options=options)) as stream:
                async for msg in stream:
                    if not isinstance(msg, ResultMessage):
                        continue
                    _raise_if_result_error(msg, f'Blacklist confirmation failed for {candidate!r}')
                    if msg.structured_output:
                        structured = msg.structured_output
        except Exception as ex:
            logger.warning(
                f'Blacklist confirmation failed for {candidate!r} ({ex}) — rejecting on the exact '
                f'name match alone.'
            )
            return label

    if structured is None:
        logger.warning(
            f'Blacklist confirmation returned no structured output for {candidate!r} — rejecting '
            f'on the exact name match alone.'
        )
        return label

    if not structured.get('same_organization'):
        logger.info(
            f'Blacklist name collision: {candidate!r} matches blacklist entry {entry_name!r} by '
            f'name but was confirmed to be a different organization — not rejecting.'
        )
        return None

    logger.info(f'Blacklisted company confirmed: {candidate!r} ({label})')
    return label


# ---------------------------------------------------------------------------
# Recruiter reposts: one role advertised under many job ids
# ---------------------------------------------------------------------------
#
# An agency re-advertises one role every few days under a new LinkedIn job id, a different country
# and a different day rate. `(site, job_id)` dedup cannot see it, the already-applied blocklist
# deliberately never holds agency names (applying once through a staffing firm must not suppress
# every other company it posts for), and the end client is usually anonymised — so one pharma
# programme produced SIX notifications between 2026-09-02 and 09-10.
#
# Text matching was measured and rejected. The extractor REWRITES each description as it condenses,
# so word-shingle containment across those six postings ran 0.62-0.86 for three of them and ~0.0 for
# the other three, while unrelated repostings by direct employers (Datadog, Pipedrive) scored 0.7+.
# No threshold separates the two cases.
#
# So: a free deterministic gate (same agency, notified inside the window), then at most one cheap
# LLM call that picks a prior job_id out of a list it was shown. It withholds a NOTIFICATION only —
# the job is still rated, saved and audited — and it fails open, because a duplicate ping costs a
# glance while a missed one costs a real role.

def raw_posting_path(candidate: dict, captured_on: date | None = None) -> Path:
    """Where this job's captured page text lives. One file per job per day."""
    captured_on = captured_on or date.today()
    site = re.sub(r'[^A-Za-z0-9_-]', '_', str(candidate.get('site') or 'unknown'))
    job_id = re.sub(r'[^A-Za-z0-9_-]', '_', str(candidate.get('job_id') or 'unknown'))
    return RAW_POSTINGS_DIR / captured_on.isoformat() / f'{site}-{job_id}.txt'


def save_raw_posting(candidate: dict, text: str, source: str, chars_sent_to_model: int) -> Path | None:
    """Keep the page an extract was made from, so its figures can be checked against it later.

    Returns the path written, or None when it could not be. A retention failure is not a run
    failure: this directory feeds nothing, and losing a capture costs a later audit, not a job.
    """
    path = raw_posting_path(candidate)
    header = yaml.safe_dump({
        'url': str(candidate.get('url') or ''),
        'job_id': str(candidate.get('job_id') or ''),
        'company': str(candidate.get('company') or ''),
        'captured': datetime.now().isoformat(timespec='seconds'),
        'source': source,
        'chars_captured': len(text),
        'chars_sent_to_model': chars_sent_to_model,
    }, sort_keys=True, allow_unicode=True)
    body = truncate_reported(
        text, RAW_POSTING_MAX_CHARS, f"raw posting capture for job {candidate.get('job_id')}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'---\n{header}---\n{body}', encoding='utf-8')
    except OSError as ex:
        logger.warning(
            f"Could not retain the raw posting for job {candidate.get('job_id')} at {path}: "
            f'{type(ex).__name__}: {ex} — continuing without it.'
        )
        return None
    return path


def prune_raw_postings(today: date | None = None) -> int:
    """Delete captured pages older than RAW_POSTINGS_RETENTION_DAYS. Returns directories removed.

    This is the only code in the repo that deletes the user's data, so it deletes exactly one
    shape of thing: a direct child of RAW_POSTINGS_DIR whose name is a date past the window. A
    child that is not a date is left alone and reported rather than guessed about.
    """
    today = today or date.today()
    if not RAW_POSTINGS_DIR.is_dir():
        return 0
    cutoff = today - timedelta(days=RAW_POSTINGS_RETENTION_DAYS)
    removed, freed = 0, 0
    for child in sorted(RAW_POSTINGS_DIR.iterdir()):
        if not child.is_dir():
            continue
        try:
            captured_on = date.fromisoformat(child.name)
        except ValueError:
            logger.warning(f'Not a capture directory, leaving it alone: {child}')
            continue
        if captured_on >= cutoff:
            continue
        # Belt and braces on a delete: resolve and confirm containment before removing anything.
        resolved = child.resolve()
        if resolved.parent != RAW_POSTINGS_DIR.resolve():
            logger.warning(f'Refusing to delete {resolved}: outside {RAW_POSTINGS_DIR}')
            continue
        freed += sum(f.stat().st_size for f in resolved.rglob('*') if f.is_file())
        try:
            shutil.rmtree(resolved)
        except OSError as ex:
            logger.warning(f'Could not prune {resolved}: {type(ex).__name__}: {ex}')
            continue
        removed += 1
    if removed:
        logger.info(
            f'Pruned {removed} raw-posting director(ies) older than {cutoff.isoformat()} '
            f'({RAW_POSTINGS_RETENTION_DAYS}-day window), freeing {freed} bytes'
        )
    return removed


RECRUITER_NOTIFICATIONS_PATH = RUN_DIR / 'recruiter_notifications.yaml'
# Above the longest description measured across 2,345 saved jobs (2026-09-24), so a repost check
# compares whole postings; at 1500 it cut 3 of this run's descriptions short.
RECRUITER_DESCRIPTION_MAX_CHARS = 8_000

_RECRUITER_NOTIFICATIONS_HEADER = (
    '# Agency postings the user has been notified about, pruned to the last\n'
    '# RECRUITER_REPOST_WINDOW_DAYS days. Used ONLY to decide whether a later posting by the same\n'
    '# agency is the same role re-advertised, and so should not be notified a second time.\n'
    '#\n'
    '# Gates nothing else: no rating, no rejection. Rewritten by the agent; safe to delete.\n'
)


def _parse_iso_date(raw: Any) -> date:
    """A `date` from an ISO string; raises ValueError otherwise. PyYAML already parses an unquoted date."""
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    try:
        return date.fromisoformat(str(raw).strip())
    except ValueError as ex:
        raise ValueError(f'{raw!r} is not an ISO YYYY-MM-DD date: {ex}') from ex


def load_recruiter_notifications(today: date | None = None) -> list[dict]:
    """Notified agency postings still inside the repost window, newest first.

    Fails open to an empty list on anything unreadable. Deliberately NOT the "skip, never rebuild"
    treatment `location_recommendations.yaml` gets: that file holds decisions the user made by hand,
    this one is rebuilt by the next notification — and an unreadable file must never be able to
    suppress a notification.
    """
    if not RECRUITER_NOTIFICATIONS_PATH.exists():
        return []
    try:
        loaded = yaml.safe_load(RECRUITER_NOTIFICATIONS_PATH.read_text(encoding='utf-8')) or []
    except Exception as ex:
        logger.warning(
            f'Could not read {RECRUITER_NOTIFICATIONS_PATH} ({unwrap_exception(ex)}) — treating it '
            f'as empty, so agency postings notify normally this run.'
        )
        return []
    if not isinstance(loaded, list):
        logger.warning(
            f'{RECRUITER_NOTIFICATIONS_PATH} must contain a YAML list, got '
            f'{type(loaded).__name__} — treating it as empty.'
        )
        return []

    cutoff = (today or date.today()) - timedelta(days=RECRUITER_REPOST_WINDOW_DAYS)
    fresh = []
    for entry in loaded:
        if not isinstance(entry, dict):
            logger.warning(f'Ignoring non-mapping entry in {RECRUITER_NOTIFICATIONS_PATH}: {snippet(entry)}')
            continue
        try:
            notified = _parse_iso_date(entry.get('notified'))
        except ValueError as ex:
            logger.warning(
                f'Ignoring entry for job {entry.get("job_id")!r} in {RECRUITER_NOTIFICATIONS_PATH}: '
                f'bad `notified` date ({ex})'
            )
            continue
        if notified >= cutoff:
            fresh.append(entry)
    fresh.sort(key=lambda e: str(e.get('notified') or ''), reverse=True)
    return fresh


def record_recruiter_notification(candidate: dict, extract: dict, today: date | None = None) -> dict:
    """Record one notified agency posting so a later repost of it can be recognised.

    Only postings actually NOTIFIED are recorded. A role first rated 3 was never sent to the user,
    so a later repost that rates 4 must still notify.
    """
    agency = str(extract.get('company') or candidate.get('company') or '').strip()
    job_id = str(candidate.get('job_id') or '')
    entry = {
        # The name as written: folding happens at the comparison, via `agency_key`.
        'agency': agency,
        'agency_key': _normalize_company(agency),
        'job_id': job_id,
        'title': str(extract.get('title') or candidate.get('title') or ''),
        'location': str(extract.get('location') or ''),
        'salary': str(extract.get('salary') or ''),
        'end_client': str(extract.get('end_client') or ''),
        'notified': (today or date.today()).isoformat(),
        'description': truncate_reported(
            str(extract.get('description') or ''), RECRUITER_DESCRIPTION_MAX_CHARS,
            f'recruiter-notification description for job {job_id}'),
    }
    entries = [e for e in load_recruiter_notifications(today=today) if str(e.get('job_id') or '') != job_id]
    entries.insert(0, entry)
    try:
        RECRUITER_NOTIFICATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RECRUITER_NOTIFICATIONS_PATH.write_text(
            _RECRUITER_NOTIFICATIONS_HEADER
            + yaml.safe_dump(entries, sort_keys=True, allow_unicode=True),
            encoding='utf-8',
        )
    except OSError as ex:
        logger.warning(f'Could not write {RECRUITER_NOTIFICATIONS_PATH}: {ex}')
    return entry


def _posting_block(posting: dict, header: str) -> str:
    return '\n'.join([
        header,
        f"job_id: {posting.get('job_id', '')}",
        f"title: {posting.get('title', '')}",
        f"location: {posting.get('location', '')}",
        f"salary: {posting.get('salary', '')}",
        f"hiring company named in the posting: {posting.get('end_client') or '(not named)'}",
        f"description: {posting.get('description', '')}",
    ])


def _repost_match_prompt(new_posting: dict, priors: list[dict]) -> str:
    prior_blocks = '\n\n'.join(
        _posting_block(p, f'--- PRIOR POSTING {i} ---') for i, p in enumerate(priors, start=1)
    )
    return (
        'A recruitment agency posted the NEW job below. The same agency already advertised the '
        'PRIOR jobs, and the user was notified about each of those.\n\n'
        f"{_posting_block(new_posting, '--- NEW POSTING ---')}\n\n{prior_blocks}\n\n"
        'Is the NEW posting the SAME ROLE as one of the prior postings — the same position at the '
        'same hiring organization or programme, re-advertised? A different country, a different '
        'day rate, or different wording in the title does NOT make it a different role: agencies '
        'routinely re-post one role across several countries. A different position, a different '
        'client, or a different programme DOES make it a different role.\n\n'
        'Judge it on the BODY: the responsibilities, the requirements, the client named or '
        'described, and concrete details such as a contact address, a rate, or a named programme. '
        'Being posted by the same agency is NOT evidence — an aggregator advertises many unrelated '
        'roles — and neither are adjacent job ids, a shared seniority level, or both postings '
        'being remote AI engineering jobs.\n\n'
        'If you are not sure, answer with an empty string. The two mistakes are not equal: a match '
        'you miss costs one duplicate notification, while a match you invent hides a real job from '
        'the user entirely.\n\n'
        'Respond with ONLY a JSON object: {"same_as_job_id": "<the job_id of the matching PRIOR '
        'POSTING above, copied exactly, or an empty string if none matches>", "reason": "<one '
        'short sentence>"}'
    )


async def _repost_match_openrouter(prompt: str, stage_stats: dict | None) -> dict:
    """Judge a repost via the OpenRouter MCP server. Raises on any failure."""
    content, cost_usd = await chat_openrouter(prompt, model=MODEL_NAME_COMPANY_MATCH)
    if stage_stats is not None:
        stage_stats['cost'] += cost_usd
    return extract_json_object(content)


async def recruiter_repost_of(
    candidate: dict, extract: dict, stage_stats: dict | None = None,
) -> dict | None:
    """The already-notified posting this one re-advertises, or None to notify normally.

    Two steps, and the first is free: unless this agency already had a posting notified inside the
    window, no LLM call is made at all — which is every job on essentially every run.

    The model only ever picks a `job_id` out of the list it was shown, and an id that is not in that
    list is treated as NO match (see the Anti-fabrication requirement): it chooses, it never
    supplies. Everything else — the window, who is compared, and what happens on a match — is code.

    Fails OPEN, the opposite of the blacklist confirmation above. A blacklist hit means a name the
    user wrote down already matched, so an outage must not readmit it; here nothing has been
    established except that the same agency posted before, and an outage must not silence a real
    role.
    """
    agency = str(extract.get('company') or candidate.get('company') or '').strip()
    if not agency:
        return None
    job_id = str(candidate.get('job_id') or '')
    agency_key = _normalize_company(agency)
    # Newest first (load_recruiter_notifications sorts), then capped: a repost follows its original
    # within days, so the newest few are the ones that can match, and an agency posting daily must
    # not grow this prompt without bound.
    priors = [
        e for e in load_recruiter_notifications()
        if e.get('agency_key') == agency_key and str(e.get('job_id') or '') != job_id
    ][:RECRUITER_REPOST_MAX_PRIORS]
    if not priors:
        return None

    new_posting = {
        'job_id': job_id,
        'title': str(extract.get('title') or candidate.get('title') or ''),
        'location': str(extract.get('location') or ''),
        'salary': str(extract.get('salary') or ''),
        'end_client': str(extract.get('end_client') or ''),
        'description': truncate_reported(
            str(extract.get('description') or ''), RECRUITER_DESCRIPTION_MAX_CHARS,
            f'repost-check description for job {job_id}'),
    }
    prompt = _repost_match_prompt(new_posting, priors)
    structured: dict | None = None

    try:
        if route_for(MODEL_NAME_COMPANY_MATCH) == 'openrouter':
            try:
                structured = await _repost_match_openrouter(prompt, stage_stats)
            except Exception as ex:
                logger.warning(
                    f'Repost check via OpenRouter failed for {agency!r}, falling back to Anthropic: '
                    f'{unwrap_exception(ex)}'
                )

        if structured is None:
            options = ClaudeAgentOptions(
                model=ANTHROPIC_MODEL_NAME_LOW,
                tools=[],
                permission_mode='bypassPermissions',
                setting_sources=[],
                strict_mcp_config=True,
                skills=[],
                output_format={
                    'type': 'json_schema',
                    'schema': {
                        'type': 'object',
                        'properties': {
                            'same_as_job_id': {
                                'type': 'string',
                                'description': (
                                    'job_id of the matching prior posting, copied exactly, or an '
                                    'empty string if none matches'
                                ),
                            },
                            'reason': {'type': 'string', 'description': 'One short sentence'},
                        },
                        'required': ['same_as_job_id', 'reason'],
                    },
                },
                cwd=str(PROJECT_DIR),
            )
            async with aclosing(sdk_query(prompt=prompt, options=options)) as stream:
                async for msg in stream:
                    if not isinstance(msg, ResultMessage):
                        continue
                    _raise_if_result_error(msg, f'Repost check failed for {agency!r}')
                    if stage_stats is not None and msg.total_cost_usd:
                        stage_stats['cost'] += msg.total_cost_usd
                    if msg.structured_output:
                        structured = msg.structured_output
    except Exception as ex:
        logger.warning(
            f'Repost check failed for {agency!r} job {job_id} ({unwrap_exception(ex)}) — '
            f'notifying anyway.'
        )
        return None

    if not structured:
        logger.warning(
            f'Repost check returned no structured output for {agency!r} job {job_id} — '
            f'notifying anyway.'
        )
        return None

    matched_id = str(structured.get('same_as_job_id') or '').strip()
    if not matched_id:
        return None
    match = next((p for p in priors if str(p.get('job_id') or '') == matched_id), None)
    if match is None:
        logger.warning(
            f'Repost check named job id {matched_id!r}, which is not one of the {len(priors)} '
            f'prior {agency!r} posting(s) it was shown — treating as no match and notifying.'
        )
        return None
    return {**match, 'reason': str(structured.get('reason') or '').strip()}


def read_pdf_pages(pdf: Path) -> list[str]:
    """Extracted text of each page of a PDF, null bytes stripped.

    Pages stay separate until a prompt serializes them (text_budget.pages_to_prompt), so size per
    page can be reported and a cap applied after whitespace normalization rather than to raw text.
    Null bytes are stripped because they make text unusable as a CLI subprocess argument.
    """
    reader = pypdf.PdfReader(pdf)
    return [(page.extract_text() or '').replace('\x00', '') for page in reader.pages]


async def _categorize_pdf_text(text: str, filename: str) -> str | None:
    """Use agent SDK to assign a category label to a PDF. Returns a snake_case string."""
    options = ClaudeAgentOptions(
        model=ANTHROPIC_MODEL_NAME_LOW,
        tools=[],
        permission_mode='bypassPermissions',
        setting_sources=[],
        strict_mcp_config=True,
        skills=[],
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
    async with aclosing(sdk_query(prompt=prompt, options=options)) as stream:
        async for msg in stream:
            if not isinstance(msg, ResultMessage):
                continue
            _raise_if_result_error(msg, f'PDF categorization failed for {filename}')
            if msg.structured_output:
                category = msg.structured_output.get('category', '')
    if category is not None:
        return underscorify(category) or 'other'
    logger.warning(f'No structured output categorizing {filename}; leaving uncategorized')
    console.print(f'[yellow]Warning: no structured output for {filename}[/yellow]')
    return None


async def categorize_save_dir_pdfs(save_dir: Path | None = None) -> None:
    """Rename uncategorized PDFs in the save directory with a cat-<category>- prefix.

    Touches the SOURCE directory only — files are renamed in place and nothing moves into the
    applied-jobs corpus here; that is ingest_save_dir_applied_pdfs()'s job.
    """
    directory = save_dir or preferences.save_dir()
    uncategorized = [p for p in directory.glob('*.pdf') if not p.name.startswith('cat-')]
    if not uncategorized:
        return
    console.print(f'[dim]Categorizing {len(uncategorized)} uncategorized PDF(s) in {directory}...[/dim]')
    for index, pdf in enumerate(uncategorized):
        try:
            pages = read_pdf_pages(pdf)
        except Exception as ex:
            logger.warning(f'Could not read {pdf}, leaving uncategorized: {type(ex).__name__}: {ex}')
            console.print(f'[yellow]Warning: could not read {pdf.name}: {ex}[/yellow]')
            continue
        console.print(f'[dim]  Categorizing {pdf.name}...[/dim]')
        prompt_text = pages_to_prompt(pages, f'PDF {pdf.name} (categorization)', PDF_PROMPT_MAX_CHARS)
        if sum(len(p) for p in pages) > PDF_PROMPT_MAX_CHARS:
            console.print(f'[yellow]Warning: {pdf.name} is over PDF_PROMPT_MAX_CHARS; see the log for sizes[/yellow]')
        try:
            category = await _categorize_pdf_text(prompt_text, pdf.name)
        except AgentApiError as ex:
            # Systemic, not per-file: the next PDF would fail identically. Retrying per file is
            # what turned one API failure into nine identical warnings and nine doomed CLI
            # subprocesses. Abort instead — leaving PDFs uncategorized means they are never
            # ingested into the applied-jobs corpus, so query generation silently degrades, and
            # that must not pass as a normal run. main() releases the run lock in its finally.
            remaining = len(uncategorized) - index
            logger.error(
                f'Aborting run: PDF categorization hit an API-level failure ({ex}). '
                f'{remaining} of {len(uncategorized)} PDF(s) left uncategorized, so they will not '
                f'enter the applied-jobs corpus.'
            )
            console.print(f'[red]Error: categorization aborted — {ex}[/red]')
            raise
        except Exception as ex:
            logger.warning(f'Categorization failed for {pdf.name}, leaving uncategorized: {ex}')
            console.print(f'[yellow]Warning: categorization failed for {pdf.name}, leaving uncategorized: {ex}[/yellow]')
            continue
        if category is None:
            continue
        new_name = f'cat-{category}-{pdf.name}'
        logger.info(f'Categorized {pdf.name} -> {new_name}')
        console.print(f'[dim]  → {new_name}[/dim]')
        pdf.rename(pdf.parent / new_name)


def _read_yaml_mapping(path: Path) -> dict:
    """Read a YAML file into a dict, warning and returning {} if it is missing or unreadable."""
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    except Exception as ex:
        logger.warning(f'Could not read YAML mapping {path}, using an empty one: {type(ex).__name__}: {ex}')
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
            logger.warning(f'Bad date prefix on {pdf}, falling back to the index/mtime: {ex}')
            console.print(f'[yellow]Warning: bad date prefix on {pdf.name}: {ex}[/yellow]')

    indexed = index.get(pdf.name, {}).get('applied_date')
    if isinstance(indexed, date):
        return indexed
    if isinstance(indexed, str):
        try:
            return datetime.strptime(indexed, '%Y-%m-%d').date()
        except ValueError as ex:
            logger.warning(f'Bad applied_date {indexed!r} for {pdf.name} in the index, falling back to mtime: {ex}')
            console.print(f'[yellow]Warning: bad applied_date for {pdf.name} in index: {ex}[/yellow]')

    legacy_mtime = legacy_cache.get(str(pdf), {}).get('mtime')
    if legacy_mtime:
        return datetime.fromtimestamp(legacy_mtime).date()

    return datetime.fromtimestamp(pdf.stat().st_mtime).date()


async def ingest_save_dir_applied_pdfs(
    save_dir: Path | None = None,
    applied_to_dir: Path | None = None,
    dry_run: bool = False,
) -> int:
    """Move cat-saved_jd-*.pdf FROM the save directory INTO the applied-jobs corpus, date-stamped.

    The two directories are separate and never defaulted from each other:
      save_dir      — source, outside the project (`save_dir` preference, default ~/Downloads)
      applied_to_dir — destination, the run_dir/applied_jobs/ corpus (APPLIED_JOBS_DIR)

    Files are renamed to '{YYYY-MM-DD}-{original name}' so the applied date survives any later
    copy that drops metadata. mtime is restored on the destination as a secondary carrier.
    Already-prefixed files are a no-op, so this is safe to run on every startup and doubles as
    the one-time migration. Returns the number of files moved (or that would move, if dry_run).
    """
    save_dir = save_dir or preferences.save_dir()
    applied_to_dir = applied_to_dir or APPLIED_JOBS_DIR

    pdfs = sorted(save_dir.glob(APPLIED_PDF_GLOB))
    if not pdfs:
        return 0

    index = _read_yaml_mapping(applied_to_dir / 'index.yaml')
    legacy_cache = _read_yaml_mapping(LEGACY_DOWNLOADS_CACHE_PATH)

    moved = 0
    for pdf in pdfs:
        applied_date = _resolve_applied_date(pdf, index, legacy_cache)
        name = pdf.name if _DATE_PREFIX_RE.match(pdf.name) else f'{applied_date.isoformat()}-{pdf.name}'
        destination = applied_to_dir / name

        if destination.exists():
            console.print(f'[yellow]Skipping {pdf.name} — {destination.name} already in the corpus.[/yellow]')
            continue

        if dry_run:
            console.print(f'[dim]  would move {pdf.name} → {destination.name}[/dim]')
            moved += 1
            continue

        mtime = pdf.stat().st_mtime
        applied_to_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(pdf), str(destination))
            os.utime(destination, (mtime, mtime))
        except Exception as ex:
            # shutil.move copies then unlinks; if the unlink failed (e.g. a read-only save dir)
            # both copies now exist and the destination-exists check would skip this file
            # forever. Roll the destination back so the next run retries cleanly.
            if pdf.exists() and destination.exists():
                try:
                    destination.unlink()
                except Exception as cleanup_ex:
                    logger.error(
                        f'{pdf} was copied to {destination} but not removed, and the partial copy '
                        f'could not be rolled back: {type(cleanup_ex).__name__}: {cleanup_ex}'
                    )
                    console.print(
                        f'[yellow]Warning: {pdf.name} was copied but not removed from '
                        f'{pdf.parent}, and the partial copy could not be rolled back: {cleanup_ex}[/yellow]'
                    )
            logger.warning(f'Could not move {pdf} to {destination}: {type(ex).__name__}: {ex}')
            console.print(f'[yellow]Warning: could not move {pdf.name}: {ex}[/yellow]')
            continue
        logger.info(f'Ingested applied job: {pdf.name} → {destination.name}')
        moved += 1

    if moved:
        verb = 'would ingest' if dry_run else 'Ingested'
        console.print(f'[dim]{verb} {moved} applied-job PDF(s) into {applied_to_dir}.[/dim]')
    return moved


async def load_applied_jobs(applied_to_dir: Path | None = None, index_path: Path | None = None) -> None:
    """Load the applied-jobs corpus, populating the in-horizon globals.

    Reads the DESTINATION directory only (run_dir/applied_jobs/) — the save directory is not
    involved here, and an un-ingested PDF still sitting in it is invisible to this function.

    index.yaml caches the extracted text and metadata per filename, keyed on mtime, so the
    metadata LLM call only runs for new or changed PDFs. Records applied to longer ago than
    APPLIED_JOBS_HORIZON_DAYS stay on disk but are not read into any global.
    """
    global _applied_companies, _reference_job_pages, _applied_jobs

    applied_to_dir = applied_to_dir or APPLIED_JOBS_DIR
    index_file = index_path or (applied_to_dir / 'index.yaml')

    index = _read_yaml_mapping(index_file)
    legacy_cache = _read_yaml_mapping(LEGACY_DOWNLOADS_CACHE_PATH)
    cutoff = date.today() - timedelta(days=APPLIED_JOBS_HORIZON_DAYS)

    companies: dict[str, str] = {}  # company_name -> pdf filename
    texts: list[tuple[date, list[str]]] = []
    jobs: list[dict] = []
    index_dirty = False
    total = 0
    aged_out = 0

    for pdf in sorted(applied_to_dir.glob('*.pdf')):
        total += 1
        applied_date = _resolve_applied_date(pdf, index, legacy_cache)
        mtime = pdf.stat().st_mtime
        entry = index.get(pdf.name)

        if entry and abs(entry.get('mtime', 0) - mtime) < INDEX_MTIME_TOLERANCE_SECONDS:
            # Entries written before per-page storage hold one joined `text`; treat it as one page.
            pages = entry.get('pages') or ([entry['text']] if entry.get('text') else [])
            metadata = {
                'company': entry.get('company', ''),
                'job_title': entry.get('job_title', ''),
                'is_agency': bool(entry.get('is_agency', False)),
                'end_client': entry.get('end_client', ''),
            }
        else:
            try:
                pages = read_pdf_pages(pdf)
            except Exception as ex:
                logger.warning(f'Could not read applied-job PDF {pdf}, skipping it: {type(ex).__name__}: {ex}')
                console.print(f'[yellow]Warning: could not read {pdf.name}: {ex}[/yellow]')
                continue

            metadata = await _extract_applied_job_metadata(
                pages_to_prompt(pages, f'PDF {pdf.name} (applied-job metadata)', PDF_PROMPT_MAX_CHARS),
                pdf.name,
            )
            index[pdf.name] = {
                'applied_date': applied_date,
                'mtime': mtime,
                'pages': pages,
                **metadata,
            }
            index_dirty = True

        if applied_date < cutoff:
            aged_out += 1
            continue

        # PDF extraction can yield null bytes; they make the text unusable as a
        # CLI subprocess argument (system prompt) — strip on load, covering
        # both fresh extractions and previously cached entries.
        pages = [page.replace('\x00', '') for page in pages]

        # A recruiting agency's own name must never enter the blocklist: applying to one role
        # through an aggregator would otherwise suppress every other company it posts for.
        if metadata['company'] and not metadata['is_agency']:
            companies[metadata['company']] = pdf.name
        if metadata['end_client']:
            companies[metadata['end_client']] = pdf.name
        if any(pages):
            texts.append((applied_date, pages))
        jobs.append({'filename': pdf.name, 'applied_date': applied_date, **metadata})

    if index_dirty:
        applied_to_dir.mkdir(parents=True, exist_ok=True)
        index_file.write_text(yaml.dump(index, default_flow_style=False, allow_unicode=True), encoding='utf-8')

    _applied_companies = companies
    # NEWEST first. Consumers cap this at MAX_REFERENCE_JOBS, and taking the OLDEST N meant a
    # newly applied job could never influence the ideal-role profile — it only entered once older
    # records aged out of the horizon, so the rater calibrated against jobs the user had moved on
    # from. Sorted by applied_date rather than left in filename order, because a legacy file
    # without a YYYY-MM-DD prefix sorts arbitrarily.
    _reference_job_pages = [pages for _date, pages in sorted(texts, key=lambda pair: pair[0], reverse=True)]
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
    path.write_text(json.dumps(record, indent=JSON_INDENT), encoding='utf-8')
    logger.info(f'Acquired run lock (pid {record["pid"]}, mode {mode})')
    return None


def release_run_lock(lock_path: Path | None = None) -> None:
    """Release the lock, but only if we are the holder — never delete another run's lock."""
    path = lock_path or RUN_LOCK_PATH
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception as ex:
        # Unreadable means the holder is unknown, so it is left in place (acquire treats it as stale).
        logger.warning(f'Run lock at {path} is unreadable, leaving it in place: {type(ex).__name__}: {ex}')
        return
    if data.get('pid') != os.getpid():
        logger.warning(f'Not releasing run lock at {path}: held by pid {data.get("pid")}, not us')
        return
    path.unlink()
    logger.info('Released run lock')


RUN_PROCESS_PATTERNS = ('job-search', 'main.py')
# A process's command line as shown in the "is a run active?" report.
RUN_PROCESS_COMMAND_DISPLAY_MAX_CHARS = 160
# `ps` normally answers instantly; this only stops a wedged process table from hanging the check.
PROCESS_SCAN_TIMEOUT_SECONDS = 10


def find_run_processes() -> list[str]:
    """Scan for job-search processes that hold no lock.

    The lock only knows about runs started after it existed, and the entry point can be either
    `main.py` or the installed `job-search` console script — so the lock alone can report
    'nothing running' while a run is very much running. This is the backstop for that.
    """
    try:
        output = subprocess.run(
            ['ps', '-axo', 'pid=,command='], capture_output=True, text=True, timeout=PROCESS_SCAN_TIMEOUT_SECONDS,
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
            matches.append(f'{pid}  {snippet(command, RUN_PROCESS_COMMAND_DISPLAY_MAX_CHARS)}')
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
        elapsed_minutes = (datetime.now() - datetime.fromisoformat(started)).total_seconds() / SECONDS_PER_MINUTE
        elapsed = f' ({elapsed_minutes:.1f} min ago)'
    except (TypeError, ValueError) as ex:
        logger.warning(f'Run lock started_at {started!r} is not an ISO timestamp, elapsed time omitted: {ex}')
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
    audit_dir = (run_dir or RUN_DIR) / 'audit_logs'
    audit_dir.mkdir(parents=True, exist_ok=True)
    stamp = (timestamp or datetime.now()).strftime('%Y%b%d-%H%M%S')
    path = audit_dir / f'audit-{stamp}.md'

    lines = [
        f'# Run audit — {(timestamp or datetime.now()).isoformat(timespec="seconds")}',
        '',
        '## 1. Applied jobs (input signal)',
        '',
        f'- In horizon ({APPLIED_JOBS_HORIZON_DAYS} days): **{applied_jobs_in_horizon}**',
        f'- Total on disk: {applied_jobs_total}',
        f'- Blocklisted companies (already applied): {len(_applied_companies)}',
        f'- Blacklisted companies (user preference, active): {len(preferences.blacklisted_companies())}',
    ]

    # An expired entry silently stops rejecting, which is exactly the kind of change that goes
    # unnoticed for months. Name it here as well as in the run log.
    if expired := preferences.expired_blacklist_entries():
        lines.append(
            '- **Expired blacklist entries (no longer rejecting):** '
            + ', '.join(f'{name} (added {added})' for name, added in expired)
        )

    # Discovery health up front: a quiet run and a broken run look identical in the tables below,
    # and that is precisely how a dead region axis survived several days unnoticed.
    health_alerts = funnel.get('health_alerts') or []
    lines += ['', '## 1b. Discovery health', '']
    if health_alerts:
        lines.append(f'**{len(health_alerts)} alert(s) — this run needs attention:**')
        lines.append('')
        lines += [f'- ⚠️ {alert}' for alert in health_alerts]
    else:
        lines.append('- No alerts: page structure sound, regions distinct, yield acceptable.')

    seen_total = funnel.get('listings_seen') or 0
    distinct_total = funnel.get('listings_distinct')
    if distinct_total is not None:
        duplicated = seen_total - distinct_total
        lines.append(
            f'- Listings inspected: {seen_total} ({distinct_total} distinct'
            + (f', {duplicated} surfaced by more than one search' if duplicated > 0 else '')
            + ')'
        )
    if overlaps := funnel.get('region_overlap'):
        lines.append('- Region overlap per query (1.0 = the location filter did nothing): '
                     + ', '.join(f'{q} {v:.0%}' for q, v in overlaps.items()))
    for report in _search_reports:
        status = 'ok'
        if report.get('blocked'):
            status = f'BLOCKED ({report["blocked"]})'
        elif report.get('violations'):
            status = 'UI CONTRACT FAILED: ' + '; '.join(report['violations'])
        elif report.get('filter_problems'):
            status = 'FILTERS NOT APPLIED: ' + '; '.join(report['filter_problems'])
        lines.append(
            f'- search `{report.get("query", "")}` / {report.get("region", "")} — '
            f'{report.get("job_cards")} cards, pin "{report.get("location_pin")}" — {status}'
        )

    # 1c. Country-list review. Advisory: nothing here changed a rating or a preference. It lives in
    # the audit log rather than only in the alert because the alert carries a count and this
    # carries the reasoning, which is what the user needs in order to promote or dismiss an entry.
    lines += ['', '## 1c. Country list review (advisory)', '']
    review: dict = {}
    if location_review.LOCATION_RECOMMENDATIONS_PATH.exists():
        try:
            review = yaml.safe_load(
                location_review.LOCATION_RECOMMENDATIONS_PATH.read_text(encoding='utf-8')
            ) or {}
        except (OSError, yaml.YAMLError) as ex:
            logger.warning(
                f'Could not read {location_review.LOCATION_RECOMMENDATIONS_PATH} for the audit '
                f'log; section 1c will list no pending countries: {type(ex).__name__}: {ex}'
            )
    pending = {
        country: entry for country, entry in (review.get('countries') or {}).items()
        if (entry or {}).get('status') == 'pending'
    }
    warnings = [w for w in (review.get('entry_warnings') or []) if (w or {}).get('status') == 'pending']
    if not pending and not warnings:
        lines.append('No pending recommendations.')
    for country, entry in sorted(pending.items()):
        lines.append(
            f'- **{country}** ({entry.get("region", "?")}) -> `{entry.get("recommendation")}` '
            f'— {entry.get("reason", "")}'
        )
    for warning in warnings:
        lines.append(
            f'- entry `{warning.get("entry")}` on `{warning.get("list")}` is a substring of '
            f'`{warning.get("collides_with")}` — {warning.get("detail", "")}'
        )
    if pending or warnings:
        lines.append('')
        lines.append(
            'Promote or dismiss these by editing `run_dir/preferences.yaml` yourself and setting '
            f'`status:` in `{location_review.LOCATION_RECOMMENDATIONS_PATH.name}`. '
            'The agent never edits preferences.'
        )

    lines += ['', '## 2. Search queries', '']

    if queries:
        lines.append(f'{len(queries)} quer(ies) generated:')
        lines.append('')
        for q in queries:
            searched = _queries_searched.get(q)
            if searched is None:
                status = '**NEVER SEARCHED**'
            elif searched == 'error':
                # Name the cause, not just the fact. Six queries once read as **FAILED** with the
                # 402 that killed all of them recorded nowhere but the run log.
                status = '**FAILED**'
                if reason := _query_errors.get(q):
                    status += f' — {snippet(reason)}'
                # A query can fail after finishing some of its searches; what it recorded first
                # is real and already queued, so say so rather than implying it found nothing.
                if partial := _check_status_per_query.get(q):
                    status += f' (after recording {format_status_counts(partial)})'
            else:
                breakdown = format_status_counts(_check_status_per_query.get(q, {}))
                status = f'{searched} listing(s) inspected ({breakdown})'
            lines.append(f'- `{q}` — {status}')
    else:
        lines.append('_No queries generated._')

    lines += ['', '## 3. Jobs found per query', '']
    if queries:
        statuses = ('new', 'already_processed', 'already_applied', 'too_old')
        lines += [
            '| Query | Listings inspected | New | Already processed | Already applied | Too old | Candidates queued |',
            '|---|---|---|---|---|---|---|',
        ]
        for q in queries:
            searched = _queries_searched.get(q)
            seen = 'never searched' if searched is None else searched
            per_status = _check_status_per_query.get(q, {})
            cells = ' | '.join(str(per_status.get(status, 0)) for status in statuses)
            lines.append(f'| {q} | {seen} | {cells} | {_candidates_per_query.get(q, 0)} |')
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
            summary = snippet((record.get('summary') or '').replace('|', '\\|').replace('\n', ' '))
            title = (record['title'] or '').replace('|', '\\|')
            company = (record['company'] or '').replace('|', '\\|')
            lines.append(
                f'| {company} | {title} | {record["outcome"]} | {rating} | {record["url"]} | {summary} |'
            )
    else:
        lines.append('_No listings inspected._')

    lines += ['', '## 5. Funnel counters', '', '```json', json.dumps(funnel, indent=JSON_INDENT), '```']

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
      - mid_rated: rated in is_mid_rated (RATINGS_MID_RATED) by the normal rater, i.e. surfaced but not acted on
    """
    pools: dict[str, list[dict]] = {'filtered': [], 'never_queued': [], 'mid_rated': []}
    for record in _listing_records.values():
        if record['check_status'] in ('too_old', 'already_applied', 'auth_required'):
            pools['filtered'].append(record)
        elif record['check_status'] == 'new' and not record['queued']:
            pools['never_queued'].append(record)
        elif is_mid_rated(record['rating']):
            pools['mid_rated'].append(record)
    return pools


def underscorify(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')


# Old MD filename pattern: job_posting-{linkedin_id}[-rating_{N}]-{company}-{desc}-{timestamp}.md
_SAVED_JOB_MD_RE = re.compile(r'^job_posting-(\d+|noid)(?:-rating_\d+)?-.+-\d+\.md$')


def load_processed_jobs() -> None:
    """Populate _processed_jobs from both old MD filenames and new per-job YAML files."""
    global _processed_jobs
    ids: set[tuple[str, str]] = set()

    # Old format: extract (linkedin, job_id) from saved MD filenames
    for f in RUN_DIR.glob('saved_jobs-*/job_posting-*.md'):
        m = _SAVED_JOB_MD_RE.match(f.name)
        if m:
            job_id = m.group(1)
            if job_id != 'noid':
                ids.add(('linkedin', job_id))

    # New format: load (site, job_id) from per-job YAML files
    for f in PROCESSED_JOBS_DIR.glob('*.yaml'):
        try:
            data = yaml.safe_load(f.read_text(encoding='utf-8'))
            if data and 'site' in data and 'job_id' in data:
                ids.add((data['site'], str(data['job_id'])))
        except Exception as ex:
            logger.warning(
                f'Could not read processed-job record {f}; that job may be re-evaluated: {type(ex).__name__}: {ex}'
            )
            console.print(f'[yellow]Warning: could not read {f}: {ex}[/yellow]')

    _processed_jobs = ids
    console.print(f'[dim]Loaded {len(_processed_jobs)} previously processed job(s).[/dim]')


def forget_processed_job(site: str, job_id: str) -> int:
    """Undo check_and_record_job for a job Stage 2 never evaluated, so the next run retries it.

    Stage 1 marks a job processed the moment it is queued. Without this, a job that Stage 2 could
    not reach (on 2026-09-21 a dead browser left 23 of them) is deduped away forever.
    Returns the number of record files removed; raises FileNotFoundError when there were none.
    """
    records = list(PROCESSED_JOBS_DIR.glob(f'job_posting-{site}-{job_id}-*.yaml'))
    if not records:
        raise FileNotFoundError(
            f'No processed-job record for {site}/{job_id} in {PROCESSED_JOBS_DIR}; '
            f'cannot release it for the next run'
        )
    for record in records:
        record.unlink()
    _processed_jobs.discard((site, job_id))
    return len(records)


EVAL_ERROR_RELEASES_PATH = RUN_DIR / 'eval_error_releases.yaml'

_EVAL_ERROR_RELEASES_HEADER = (
    '# How many times each job was released for another run after its Stage 2 evaluation errored.\n'
    '# At EVAL_ERROR_MAX_RELEASES a job is no longer released. Rewritten by the agent; safe to delete.\n'
)


def release_errored_job(site: str, job_id: str) -> str:
    """Release a job whose evaluation errored so the next run rates it; returns what was done, for the log.

    Stage 1 marks a job processed when it is queued, so an error in Stage 2 used to end the job for
    good: a posting the rater had scored 5 was lost on 2026-09-03 and again on 2026-10-08. The
    count is kept on disk and capped, because a posting that fails every time would otherwise be
    opened on the real account on every run.

    Raises when the count file cannot be read or written, or the job has no processed record: the
    caller is already reporting an error for this job and adds this one to it.
    """
    key = f'{site}/{job_id}'
    releases: dict = {}
    if EVAL_ERROR_RELEASES_PATH.exists():
        loaded = yaml.safe_load(EVAL_ERROR_RELEASES_PATH.read_text(encoding='utf-8')) or {}
        if not isinstance(loaded, dict):
            raise ValueError(
                f'{EVAL_ERROR_RELEASES_PATH} must contain a YAML mapping of job to release count, '
                f'got {type(loaded).__name__}; {key} was not released'
            )
        releases = loaded
    released_before = int(releases.get(key, 0))
    if released_before >= EVAL_ERROR_MAX_RELEASES:
        return (f'NOT retried: already released {released_before} time(s) '
                f'(EVAL_ERROR_MAX_RELEASES), so it stays in processed_jobs/')
    # Count first: if the write fails the job stays processed, which is the old behaviour, rather
    # than released with no record of it.
    releases[key] = released_before + 1
    EVAL_ERROR_RELEASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_ERROR_RELEASES_PATH.write_text(
        _EVAL_ERROR_RELEASES_HEADER + yaml.safe_dump(releases, sort_keys=True), encoding='utf-8')
    forget_processed_job(site, job_id)
    return f'released for the next run (release {released_before + 1} of {EVAL_ERROR_MAX_RELEASES})'


# --- Tool implementations (plain async functions, directly testable) ---

async def do_save_job_posting(
    company: str, description: str, rating: int, content: str, job_id: str | None = None
) -> dict:
    date_str = datetime.now().strftime('%Y%b%d')
    dir_path = RUN_DIR / f'saved_jobs-{date_str}'
    dir_path.mkdir(parents=True, exist_ok=True)

    ts = int(time.time())
    id_part = job_id if job_id else 'noid'
    rating_part = f'-rating_{rating}' if rating is not None else ''
    # Cap name components (SAVED_JOB_FILENAME_*); the full text is in the file itself.
    company_part = underscorify(company)[:SAVED_JOB_FILENAME_COMPANY_MAX_CHARS].rstrip('_')
    description_part = underscorify(description)[:SAVED_JOB_FILENAME_DESCRIPTION_MAX_CHARS].rstrip('_')
    filename = f'job_posting-{id_part}{rating_part}-{company_part}-{description_part}-{ts}.md'
    (dir_path / filename).write_text(content, encoding='utf-8')

    return {'content': [{'type': 'text', 'text': f'Saved: saved_jobs-{date_str}/{filename}'}]}


async def do_update_job_requirements(content: str) -> dict:
    JOB_REQUIREMENTS_PATH.write_text(content, encoding='utf-8')
    return {
        'content': [
            {'type': 'text', 'text': f'JOB_REQUIREMENTS.md updated. New contents:\n\n{content}'}
        ]
    }


COST_LOG_PATH = RUN_DIR / 'cost_log.jsonl'


def log_run_cost(record: dict, log_path: Path | None = None) -> None:
    """Append a per-run cost record as one JSON line to cost_log.jsonl."""
    path = log_path if log_path is not None else COST_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')
    # Elapsed and cost otherwise reach only this file and Telegram, never the run log.
    if 'total_cost' in record and 'elapsed_minutes' in record:
        stage_costs = ', '.join(
            f"{stage} ${stats['cost']:.4f}"
            for stage, stats in (record.get('stage_stats') or {}).items() if stats.get('cost')
        )
        logger.info(
            f"Run complete ({record.get('status')}): {record['elapsed_minutes']:.1f} min, "
            f"${record['total_cost']:.4f} total ({stage_costs or 'no stage cost'})"
        )


# Written as the name is written, like every other place name (see docs/requirements.md, Names
# stay proper); `re.IGNORECASE` below is what makes it match "the us" too.
_US = r'(?:US|U\.S\.|United States)'

_AUTH_REQUIRED_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        rf'must be (legally )?authorized to work in the {_US}',
        r'no visa sponsorship',
        r'not able to (provide |offer )?sponsor',
        r'cannot (provide |offer )?sponsor',
        r'sponsorship (is )?not (available|provided|offered)',
        r'will not (provide |offer )?sponsor',
        r'unable to (provide |offer )?sponsor',
        rf'{_US} citizens? and (lawful )?permanent residents?',
        rf'currently (legally )?authorized to work in the {_US}',
        r"this (position|role|job) (does not|doesn't) (provide|offer|support) (visa )?sponsorship",
        r'employment authorization (without|not requiring) sponsorship',
    ]
]


def _requires_current_us_auth(text: str) -> bool:
    """Return True if text indicates the job requires current US work authorization (no sponsorship)."""
    return any(p.search(text) for p in _AUTH_REQUIRED_PATTERNS)


# Both are SEARCHED within the string, not matched against all of it. LinkedIn and the extract
# model write "Reposted 4 days ago", "2026-09-22 (3 days ago)" and "Posted 19 hours ago (viewed
# 2026-09-24)"; anchoring on the bare forms produced 98 unparsed-date warnings across logged runs, and a
# rejected date silently skips the stale-posting rule (2026-09-24).
# A LEADING date is the posting date; a date later in the string is often when the page was viewed.
_LEADING_ISO_DATE_RE = re.compile(r'^(?P<iso>\d{4}-\d{2}-\d{2})\b')
_RELATIVE_DATE_RE = re.compile(r'\b(?P<count>\d+)\s+(?P<unit>minute|hour|day|week|month)s?\s+ago\b')
_JUST_POSTED_PHRASES = ('just now', 'today', 'moments ago')
# "N months ago" is approximate by nature; a calendar-exact month would imply false precision.
DAYS_PER_MONTH_APPROX = 30


def parse_posting_date(date_posted: str | None) -> date | None:
    """Parse an absolute (YYYY-MM-DD) or relative ('4 days ago') posting date.

    A leading YYYY-MM-DD wins; otherwise the first "N units ago" anywhere in the string, so
    prefixes ("Reposted") and trailing notes ("(viewed 2026-09-24)") are tolerated.

    None when no date was given. Raises ValueError for a non-empty string in no known format, so
    the caller, which knows the job, decides how to report it.
    """
    if not date_posted:
        return None
    s = date_posted.strip().lower()
    if iso := _LEADING_ISO_DATE_RE.match(s):
        try:
            return date.fromisoformat(iso.group('iso'))
        except ValueError as ex:
            raise ValueError(f'posting date {date_posted!r} looks ISO but is not a real date: {ex}') from ex
    m = _RELATIVE_DATE_RE.search(s)
    if m:
        n, unit = int(m.group('count')), m.group('unit')
        deltas = {'minute': timedelta(minutes=n), 'hour': timedelta(hours=n), 'day': timedelta(days=n),
                  'week': timedelta(weeks=n), 'month': timedelta(days=n * DAYS_PER_MONTH_APPROX)}
        return (datetime.now() - deltas[unit]).date()
    if s in _JUST_POSTED_PHRASES:
        return date.today()
    raise ValueError(
        f'posting date {date_posted!r} is neither YYYY-MM-DD, "N minutes/hours/days/weeks/months ago", '
        f'nor one of {_JUST_POSTED_PHRASES}'
    )


# Outcomes record_listings reports on one line per search, carrying every job id, instead of one
# line per listing: both mean "seen before, nothing to decide", and on a saturated search they
# are most of the page.
RECORD_LISTINGS_FOLDED_STATUSES = ('already_processed', 'already_applied')


async def do_check_and_record_job(
    site: str, job_id: str, company: str, description: str,
    date_posted: str | None = None,
    url: str | None = None, content: str | None = None,
    log_folded_statuses: bool = True,
) -> dict:
    def _result(status: str) -> dict:
        _check_status_counts[status] = _check_status_counts.get(status, 0) + 1
        # Distinct-listing bookkeeping, kept separate from the CALL counts above. Two regions
        # returning the same 25 cards must not read as 50 listings of coverage -- that is what hid
        # the dead EU region and what pushed a collapsed query away from the low-yield retry.
        _distinct_listing_ids.add(f'{site}/{job_id}')
        _search_ids.setdefault((_current_query or '', _current_region or ''), set()).add(job_id)
        _listing_records[(site, job_id)] = {
            'site': site, 'job_id': job_id, 'company': company, 'title': description,
            'date_posted': date_posted, 'check_status': status,
            'url': url or job_url(site, job_id),
            'queued': False, 'outcome': status, 'rating': None, 'summary': '',
        }
        # record_listings reports a search's folded outcomes on one line each of its own.
        quiet = status in RECORD_LISTINGS_FOLDED_STATUSES and not log_folded_statuses
        logger.log(logging.DEBUG if quiet else logging.INFO,
                   f'check_and_record_job: {status} — {company} — {description} '
                   f'[{site}/{job_id}, date_posted={date_posted!r}]')
        return {'content': [{'type': 'text', 'text': status}]}

    key = (site, job_id)
    if key in _processed_jobs:
        return _result('already_processed')

    matched_pdf = await company_matches_applied(company)
    if matched_pdf is not None:
        console.print(f'[dim]Skipping {company} — already applied ({matched_pdf}).[/dim]')
        return _result('already_applied')

    try:
        posted = parse_posting_date(date_posted)
    except ValueError as ex:
        # Unknown age is not a reason to skip; the Stage 2 stale rule gets another look.
        logger.warning(f'{site} job {job_id} ({company}): {ex} — treating its age as unknown')
        posted = None
    if posted and (date.today() - posted).days > JOB_MAX_AGE_DAYS:
        return _result('too_old')

    combined_text = ' '.join(filter(None, [description, content]))
    if _requires_current_us_auth(combined_text):
        console.print(f'[dim]Skipping {company} — requires current US work authorization.[/dim]')
        return _result('auth_required')

    PROCESSED_JOBS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime('%Y%b%d')
    ts = int(time.time())
    filename = f'job_posting-{site}-{job_id}-{date_str}-{ts}-{underscorify(company)}-{underscorify(description)}.yaml'
    data = {
        'site': site,
        'job_id': job_id,
        'date_posted': date_posted,
        'date_recorded': date.today().isoformat(),
        'company': company,
        'description': description,
    }
    if url:
        data['url'] = url
    if content:
        data['content'] = content
    (PROCESSED_JOBS_DIR / filename).write_text(yaml.dump(data, default_flow_style=False), encoding='utf-8')
    _processed_jobs.add(key)
    return _result('new')


async def do_submit_job_extract(
    title: str, company: str, description: str,
    location: str | None = None, date_posted: str | None = None,
    closed: bool = False, salary: str | None = None, sponsorship_note: str | None = None,
    language_requirement: str | None = None, relocation: str | None = None,
    residency_scope: str | None = None,
    workplace_type: str | None = None, education_requirement: str | None = None,
    is_agency: bool | None = None, end_client: str | None = None,
    posting_language: str | None = None, stated_working_language: str | None = None,
) -> dict:
    _job_extracts.append({
        'title': title, 'company': company, 'description': description,
        'location': location or '', 'date_posted': date_posted or '',
        'closed': closed, 'salary': salary or '', 'sponsorship_note': sponsorship_note or '',
        'language_requirement': language_requirement or '', 'relocation': relocation or '',
        'residency_scope': (residency_scope or '').strip().lower(),
        'posting_language': (posting_language or '').strip().lower(),
        'stated_working_language': (stated_working_language or '').strip().lower(),
        'workplace_type': (workplace_type or '').strip().lower(),
        'education_requirement': (education_requirement or '').strip().lower(),
        # None (not False) when the extractor said nothing, so derive_agency_posting() can tell
        # "the model judged this not an agency" from "the model did not answer".
        'is_agency': is_agency,
        'end_client': (end_client or '').strip(),
    })
    return {'content': [{'type': 'text', 'text': 'Extract submitted.'}]}


async def do_queue_candidate(
    site: str, job_id: str, url: str, title: str, company: str,
    snippet: str, date_posted: str | None = None, query: str | None = None,
) -> dict:
    if (reason := title_rejection_reason(title)):
        _queue_skipped_counts[reason] = _queue_skipped_counts.get(reason, 0) + 1
        if (record := _listing_records.get((site, job_id))) is not None:
            record['outcome'] = 'queue_skipped'
            record['query'] = query
            record['url'] = url or record['url']
            record['summary'] = reason
        logger.info(f'queue_candidate: SKIPPED {company} — {title} [{site}/{job_id}]: {reason}')
        return {'content': [{'type': 'text', 'text': f'Skipped (not queued): {reason}. Continue with the next listing.'}]}
    _candidates.append({
        'site': site, 'job_id': job_id, 'url': url,
        'title': title, 'company': company,
        'date_posted': date_posted or '', 'snippet': snippet,
    })
    # Attribute to the query run_scraper is actually on, not the string the model echoed back.
    # Since the region and filter words moved into the search text, the model passes the full
    # search string ("Principal AI Engineer, remote, Canada, senior level") while run_scraper
    # keys everything on the base query ("Principal AI Engineer"). The keys never matched, so
    # every per-query candidate count read 0. Attribution is the caller's to know, not the
    # model's to remember.
    attributed = _current_query or query
    if attributed:
        _candidates_per_query[attributed] = _candidates_per_query.get(attributed, 0) + 1
    else:
        logger.warning(f'queue_candidate called with no active query for {company} — {title}; '
                       'per-query counts will be understated')
    record = _listing_records.get((site, job_id))
    if record is not None:
        record['queued'] = True
        record['query'] = attributed
        record['url'] = url or record['url']
        record['snippet'] = snippet
        record['outcome'] = 'queued'
    logger.info(f'queue_candidate: {company} — {title} [{site}/{job_id}, query={query!r}]')
    return {'content': [{'type': 'text', 'text': f'Queued: {company} — {title}'}]}


async def do_record_listings(jobs: list[dict], query: str | None = None) -> str:
    """Record a whole search's harvested listings in one call, from CODE's copy of the harvest.

    `jobs` never originates in the model. This is the anti-fabrication guarantee: a model asked to
    relay a payload cannot tell copying from producing, so when it has no data it emits a plausible
    object instead of failing (measured 2026-08-21: an evaluate result was diverted to a file and
    the model invented a whole page report). Removing the argument removes the opportunity, which
    is not something a prompt can do. See the Anti-fabrication requirement in AGENTS.md.

    Also collapses ~25 per-listing tool calls into one, and enforces
    SCRAPER_MAX_LISTINGS_PER_SEARCH in code rather than as a prompt suggestion.

    Returns a SHORT summary for the model: counts only, never job data.
    """
    usable = [j for j in jobs if isinstance(j, dict) and str(j.get('id') or '').strip()]
    if (malformed := len(jobs) - len(usable)):
        logger.warning(f'record_listings: dropped {malformed} harvested entr(ies) with no job id')
    if not usable:
        return ('nothing to record: the harvest produced no usable listings. Do NOT describe or '
                'invent listings — say what you saw and stop.')

    capped = usable[:SCRAPER_MAX_LISTINGS_PER_SEARCH]
    before_skips = sum(_queue_skipped_counts.values())
    tally: dict[str, int] = {}
    folded_ids: dict[str, list[str]] = {}
    queued = 0
    for job in capped:
        job_id = str(job.get('id'))
        status_result = await do_check_and_record_job(
            'linkedin', job_id, str(job.get('company') or ''), str(job.get('title') or ''),
            date_posted=str(job.get('posted') or '') or None,
            log_folded_statuses=False,
        )
        status = status_result['content'][0]['text'].strip()
        tally[status] = tally.get(status, 0) + 1
        if status in RECORD_LISTINGS_FOLDED_STATUSES:
            folded_ids.setdefault(status, []).append(job_id)
        if status == 'new':
            await do_queue_candidate(
                'linkedin', job_id, job_url('linkedin', job_id), str(job.get('title') or ''),
                str(job.get('company') or ''), str(job.get('location') or ''),
                date_posted=str(job.get('posted') or '') or None, query=query,
            )
            queued += 1
    skipped = sum(_queue_skipped_counts.values()) - before_skips
    cap_note = f' (capped from {len(usable)})' if len(usable) > len(capped) else ''
    parts = ', '.join(f'{count} {status}' for status, count in sorted(tally.items()))
    for status in RECORD_LISTINGS_FOLDED_STATUSES:
        if (ids := folded_ids.get(status)):
            # Every id, never a sample: this line is what tells seen-but-deduped from never-seen.
            logger.info(f'{status}: {len(ids)}/{len(capped)} linkedin={",".join(ids)}')
    logger.info(f'record_listings: {len(capped)}{cap_note} recorded — {parts}; '
                f'{queued} queued, {skipped} skipped by title')
    return (f'recorded {len(capped)}{cap_note}: {parts}. {queued} queued, '
            f'{skipped} skipped by title. Continue with the next search.')


# --- Tool wrappers (SDK @tool decorators delegate to the implementations above) ---

@tool(
    'save_job_posting',
    'Save a job posting to the daily saved_jobs directory. Call this for every job evaluated. '
    'Extract the LinkedIn job ID from the URL (e.g. linkedin.com/jobs/view/1234567890/) and pass it as job_id.',
    # Explicit JSON Schema, not the {"name": str} shorthand: that shorthand marks EVERY key
    # required (claude_agent_sdk/__init__.py:422), and a required job_id makes this tool
    # uncallable for a posting whose id is not visible — the model either refuses or invents one.
    {
        'type': 'object',
        'properties': {
            'company': {'type': 'string'},
            'description': {'type': 'string'},
            'rating': {'type': 'integer'},
            'content': {'type': 'string'},
            'job_id': {'type': 'string', 'description': 'LinkedIn job id if visible; omit if not'},
        },
        'required': ['company', 'description', 'rating', 'content'],
    },
)
async def save_job_posting(args: dict[str, Any]) -> dict:
    return await do_save_job_posting(
        args['company'], args['description'], args['rating'], args['content'],
        job_id=args.get('job_id'),
    )


@tool(
    'update_job_requirements',
    "Rewrite JOB_REQUIREMENTS.md with a complete, updated summary of the user's job preferences. "
    "Always rewrite the full file — never append. Returns the new contents so they are in context.",
    {'content': str},
)
async def update_job_requirements(args: dict[str, Any]) -> dict:
    return await do_update_job_requirements(args['content'])


@tool(
    'check_and_record_job',
    "Before evaluating any job, call this with the site name, job ID, company name, and job title/description. "
    "Returns 'already_processed' (skip it), 'too_old' (skip it), 'already_applied' (skip it), "
    "'auth_required' (skip it — requires current US work authorization), or 'new' (proceed to evaluate). "
    "date_posted is optional — pass whatever is visible (YYYY-MM-DD or relative like '4 days ago'); omit if not shown. "
    "Optionally pass url (the job posting URL) and content (full text of the posting) to persist them in the record.",
    {
        'type': 'object',
        'properties': {
            'site': {'type': 'string'},
            'job_id': {'type': 'string'},
            'company': {'type': 'string'},
            'description': {'type': 'string'},
            'date_posted': {'type': 'string', 'description': 'YYYY-MM-DD or relative; omit if not shown'},
            'url': {'type': 'string'},
            'content': {'type': 'string'},
        },
        'required': ['site', 'job_id', 'company', 'description'],
    },
)
async def check_and_record_job(args: dict[str, Any]) -> dict:
    return await do_check_and_record_job(
        args['site'], args['job_id'], args['company'], args['description'],
        date_posted=args.get('date_posted'), url=args.get('url'), content=args.get('content'),
    )


@tool(
    'queue_candidate',
    "Add a job candidate to the internal evaluation queue. "
    "Call this after check_and_record_job returns 'new'. "
    "Pass what is visible in the search results: URL, title, company, snippet. "
    "date_posted is optional — pass it if visible (exact or relative), omit if not shown. "
    "query is optional — pass the search query string that returned this result (e.g. 'Staff ML Engineer'). "
    "Do NOT navigate to the individual job page — a separate agent handles that in stage 2.",
    {
        'type': 'object',
        'properties': {
            'site': {'type': 'string'},
            'job_id': {'type': 'string'},
            'url': {'type': 'string'},
            'title': {'type': 'string'},
            'company': {'type': 'string'},
            'snippet': {'type': 'string'},
            'date_posted': {'type': 'string', 'description': 'exact or relative; omit if not shown'},
            'query': {'type': 'string', 'description': 'search query that returned this result'},
        },
        'required': ['site', 'job_id', 'url', 'title', 'company', 'snippet'],
    },
)
async def queue_candidate(args: dict[str, Any]) -> dict:
    return await do_queue_candidate(
        args['site'], args['job_id'], args['url'], args['title'],
        args['company'], args['snippet'], date_posted=args.get('date_posted'),
        query=args.get('query'),
    )


@tool(
    'submit_job_extract',
    "Submit the condensed extract of the job posting page you navigated to. "
    "Include only information-dense content: requirements, responsibilities, stack, seniority — "
    "strip navigation chrome, boilerplate, and similar-jobs lists. "
    "Set closed=true if the page shows 'No longer accepting applications' or equivalent. "
    "Pass date_posted exactly as shown (absolute or relative like '4 days ago'). "
    "Pass sponsorship_note with any visa/work-authorization statement, verbatim, if present. "
    "Pass language_requirement with languages explicitly REQUIRED (not nice-to-have), comma-separated "
    "lowercase, e.g. 'english, german'; omit if no language requirement is stated. "
    "Pass posting_language with the language the SOURCE PAGE ITSELF IS WRITTEN IN, lowercase, e.g. "
    "'english', 'french'. Judge the original page you read, NOT the condensed English text you are "
    "writing here — you translate as you condense, so your own output says nothing about the "
    "original. The original job title is usually the clearest tell (a title like 'Scientifique "
    "principal des données en IA' means 'french'). Omit only if genuinely undeterminable. "
    "Pass stated_working_language with the language the posting SAYS the team works in "
    "('our working language is English'); omit it unless stated. "
    "Pass relocation with the country/city if the posting requires relocating to or residing in a "
    "specific place (e.g. 'must be based in Portugal'); omit for work-from-anywhere roles. "
    "Pass residency_scope as 'country_only' if the posting requires LIVING IN the country it "
    "is advertised in ('Remote within country', 'must be based in Germany', 'open only to "
    "candidates residing in Poland'), or 'area_wide' if it offers a whole multi-country area "
    "('remote anywhere in the EU', 'Work from Anywhere', 'any EMEA country'); omit it when the "
    "posting does not say. This is where the HOLDER MUST LIVE, not where the job is "
    "advertised \u2014 'Romania (Remote)' on its own says nothing here. "
    "Pass workplace_type as exactly 'remote', 'hybrid', or 'onsite' when the page states the work "
    "arrangement; omit only if the page genuinely does not say. Any mention of required days in "
    "the office (e.g. '2-3 days onsite') is 'hybrid', not 'remote'. "
    "Pass education_requirement as 'master' or 'phd' ONLY if the posting states an advanced degree "
    "as a hard requirement (e.g. 'MSc in Computer Science required', 'PhD is a must'); omit it when "
    "the degree is merely preferred, when equivalent experience is accepted (\"Master's or equivalent "
    "practical experience\", 'MSc a plus', \"Bachelor's or Master's\"), or when only a Bachelor's is required. "
    "Pass is_agency=true if the poster is a staffing firm, recruiting agency, or job aggregator "
    "reposting on behalf of another company rather than the employer that would actually hire, and "
    "pass end_client with that hiring company's name if the posting names it (agencies usually keep "
    "it anonymous, e.g. 'our client, a leading fintech' — leave end_client empty in that case).",
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
            'salary': {'type': 'string', 'description': SALARY_FIELD_DESCRIPTION},
            'sponsorship_note': {'type': 'string'},
            'language_requirement': {
                'type': 'string',
                'description': "Explicitly required languages, comma-separated lowercase, e.g. 'english, german'",
            },
            'posting_language': {
                'type': 'string',
                'description': "Language the SOURCE page is written in, lowercase e.g. 'english', 'french' — "
                               'judge the original page, not your condensed English output; '
                               'the original title is the clearest tell',
            },
            'stated_working_language': {'type': 'string', 'description': STATED_WORKING_LANGUAGE_FIELD_DESCRIPTION},
            'residency_scope': {'type': 'string', 'enum': ['country_only', 'area_wide', ''],
                                'description': "Whether the posting pins residence to the country it is anchored in "
                                               "('country_only') or offers a whole multi-country area ('area_wide'); "
                                               'empty when the posting does not say'},
            'relocation': {'type': 'string', 'description': 'Location the candidate must relocate to / reside in, if the posting requires one'},
            'education_requirement': {
                'type': 'string',
                'description': "'master' or 'phd' if an advanced degree is a HARD requirement; "
                               'empty when merely preferred or when equivalent experience is accepted',
            },
            # Wording mirrors _extract_applied_job_metadata's is_recruiting_agency/end_client_name
            # so the applied-job and scraped sides classify a poster identically.
            'is_agency': {
                'type': 'boolean',
                'description': (
                    'True if the poster is a staffing firm, recruiting agency, or job aggregator '
                    'reposting on behalf of another company, rather than the employer that '
                    'would actually hire.'
                ),
            },
            'end_client': {
                'type': 'string',
                'description': 'The company that would actually hire, if the posting names one; otherwise an empty string',
            },
        },
        'required': ['title', 'company', 'description'],
    },
)
async def submit_job_extract(args: dict[str, Any]) -> dict:
    return await do_submit_job_extract(
        args['title'], args['company'], args['description'],
        location=args.get('location'), date_posted=args.get('date_posted'),
        closed=args.get('closed', False), salary=args.get('salary'),
        sponsorship_note=args.get('sponsorship_note'),
        language_requirement=args.get('language_requirement'), relocation=args.get('relocation'),
        residency_scope=args.get('residency_scope'),
        workplace_type=args.get('workplace_type'),
        education_requirement=args.get('education_requirement'),
        is_agency=args.get('is_agency'),
        end_client=args.get('end_client'),
        posting_language=args.get('posting_language'),
        stated_working_language=args.get('stated_working_language'),
    )


# --- MCP server factories ---

def make_job_search_server(interactive: bool):
    """MCP server for interactive mode."""
    tools = [check_and_record_job, save_job_posting]
    if interactive:
        tools.append(update_job_requirements)
    return create_sdk_mcp_server(name='job_search', version='1.0.0', tools=tools)


# --- UI contract -------------------------------------------------------------------------------


def _fingerprint_path() -> Path:
    return RUN_DIR / UI_FINGERPRINT_FILENAME


def detect_block_signature(report: dict) -> str:
    """Return the block phrase found in the page, or '' if none.

    Checked BEFORE any contract judgement, because the two need opposite responses: a moved
    selector may be retried, a challenge must never be. Pushing through a block is what turns a
    soft suspicion into a confirmed evasion pattern, and a banned account ends the job search.
    """
    haystack = ' '.join(str(report.get(key, '')) for key in ('body_sample', 'result_count_text'))
    haystack = haystack.lower()
    for phrase in UI_BLOCK_SIGNATURES:
        if phrase in haystack:
            return phrase
    return ''


def evaluate_ui_contract(report: dict) -> list[str]:
    """Return a list of contract violations; empty means the page looks structurally sound.

    Deliberately code-side. Asking the model 'does this page look right?' is the same discretion
    that let a dead region and a restructured results list run for days -- it reports, code judges.
    """
    violations: list[str] = []
    for element in UI_CONTRACT_ELEMENTS:
        if not element['required']:
            continue
        name = element['name']
        value = report.get(name)
        missing = (value in (None, '', False)) or (isinstance(value, int) and value == 0)
        if missing:
            violations.append(f"{name} missing ({element['description']})")

    # Zero cards while the page itself claims results is the specific silent-failure shape:
    # the harvest returns nothing and the run reports "no new jobs today".
    count_text = str(report.get('result_count_text', ''))
    if not report.get('job_cards') and re.search(r'\d', count_text):
        violations.append(f'0 job cards harvested while the page shows "{count_text.strip()}"')
    return violations


def check_filters_applied(report: dict, region: dict) -> list[str]:
    """Return reasons the requested filters do NOT look applied.

    This is where the geoId/f_TPR discovery earns its keep. We never NAVIGATE to those parameters
    -- that is the bot-shaped move -- but once the chips are clicked LinkedIn puts them in the URL
    itself, so reading them back proves the clicks landed. Before this existed, a location filter
    that silently did nothing was invisible.
    """
    problems: list[str] = []
    url = str(report.get('url', ''))
    chips = [str(c).lower() for c in (report.get('chips') or [])]
    pin = str(report.get('location_pin', '')).strip().lower()

    wanted_location = str(region.get('linkedin_location', '')).strip().lower()
    if wanted_location:
        geo_id = str(region.get('geo_id', '')).strip()
        if geo_id and f'geoId={geo_id}' not in url:
            problems.append(f'location: geoId={geo_id} absent from the URL')
        elif not geo_id and wanted_location not in pin:
            problems.append(f'location: pin reads "{report.get("location_pin", "")}", wanted "{wanted_location}"')

    if f'f_TPR=r{SCRAPER_DATE_POSTED_SECONDS}' not in url and SCRAPER_DATE_POSTED_LABEL.lower() not in chips:
        problems.append(f'date posted: neither f_TPR=r{SCRAPER_DATE_POSTED_SECONDS} nor a "{SCRAPER_DATE_POSTED_LABEL}" chip')

    # Experience level applies server-side with no URL parameter, so the relabelled chip is the
    # only evidence available.
    if SCRAPER_EXPERIENCE_LABEL.lower() not in chips:
        problems.append(f'experience level: no "{SCRAPER_EXPERIENCE_LABEL}" chip')
    return problems


_CHIP_OF_CHOSEN_VALUE = {value: chip for chip, values in UI_CHIP_CHOSEN_VALUES.items() for value in values}


def _structural_chips(labels: list | None) -> list[str]:
    '''The structural chips among `labels`, each under its own name whatever value is chosen.'''
    return sorted({_CHIP_OF_CHOSEN_VALUE.get(str(c), str(c)) for c in labels or []
                   if str(c) in UI_STRUCTURAL_CHIPS})


def check_fingerprint_drift(report: dict) -> str:
    """Compare the page shape against the last known good one; returns a description or ''.

    Fires even when every required element is still present, so a restructure is noticed the run
    it happens rather than months later when something finally breaks.
    """
    # Structural chips only: the topical suggestions vary per query by design, and including
    # them made this fire on every single query.
    observed = _structural_chips(report.get('chips'))
    path = _fingerprint_path()
    previous: list[str] = []
    if path.exists():
        try:
            stored = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
            # Filtered like `observed`, so a chip later dropped from UI_STRUCTURAL_CHIPS does not
            # read as "gone" against a fingerprint written before it was dropped.
            previous = _structural_chips(stored.get('chips'))
        except Exception as ex:
            logger.warning(f'UI fingerprint at {path} unreadable, treating as absent: {ex}')

    if observed and observed != previous:
        try:
            path.write_text(
                yaml.safe_dump({'chips': observed, 'updated': datetime.now().isoformat()},
                               allow_unicode=True, sort_keys=False),
                encoding='utf-8',
            )
        except Exception as ex:
            logger.warning(f'could not write UI fingerprint to {path}: {ex}')
        if previous:
            added = [c for c in observed if c not in previous]
            removed = [c for c in previous if c not in observed]
            parts = []
            if added:
                parts.append(f'new chips {added}')
            if removed:
                parts.append(f'chips gone {removed}')
            return '; '.join(parts) if parts else ''
    return ''


@tool(
    'report_search',
    'Report the structural state of one search results page (from the UI-contract evaluate call).',
    {
        'query': str,
        'region': str,
        'report': dict,
    },
)
async def report_search(args: dict) -> dict:
    """Record one search and judge whether the page is usable. Judgement lives here, not in the prompt."""
    query = str(args.get('query') or _current_query or '')
    region_name = str(args.get('region') or '')
    report = args.get('report') or {}
    if not isinstance(report, dict):
        return {'content': [{'type': 'text', 'text': 'report must be the object returned by the contract evaluate call'}]}

    global _current_region
    _current_region = region_name

    regions = {str(r.get('name') or r.get('linkedin_location', '')): r for r in preferences.search_regions()}
    region = regions.get(region_name, {})

    blocked = detect_block_signature(report)
    violations = [] if blocked else evaluate_ui_contract(report)
    filter_problems = [] if (blocked or violations) else check_filters_applied(report, region)
    drift = check_fingerprint_drift(report) if not blocked else ''

    record = {
        'query': query,
        'region': region_name,
        'job_cards': report.get('job_cards'),
        'location_pin': report.get('location_pin'),
        'result_count_text': report.get('result_count_text'),
        'blocked': blocked,
        'violations': violations,
        'filter_problems': filter_problems,
        'drift': drift,
    }
    _search_reports.append(record)

    # A passing report is DEBUG: every failing branch below logs its own WARNING.
    logger.debug(
        f'report_search: query="{query}" region="{region_name}" cards={report.get("job_cards")} '
        f'pin="{report.get("location_pin")}" blocked={blocked or "no"} '
        f'violations={len(violations)} filter_problems={len(filter_problems)}'
    )

    if blocked:
        _ui_alerts.append({'kind': 'blocked', 'query': query, 'region': region_name, 'detail': blocked})
        logger.warning(f'Stage 1b BLOCKED on "{query}" / {region_name}: page matched "{blocked}"')
        return {'content': [{'type': 'text', 'text':
            f'BLOCKED: the page matched "{blocked}". Stop this query now. Do NOT retry it, do not '
            'reload, and do not try to work around it. Say what you saw and move on.'}]}

    if violations:
        _ui_alerts.append({'kind': 'contract', 'query': query, 'region': region_name,
                           'detail': '; '.join(violations)})
        logger.warning(f'Stage 1b UI CONTRACT broken on "{query}" / {region_name}: {"; ".join(violations)}')
        return {'content': [{'type': 'text', 'text':
            'UI CONTRACT FAILED: ' + '; '.join(violations) + '. LinkedIn has changed this page. '
            'Stop this query and say exactly what you saw. Do not fall back to clicking the '
            'results list.'}]}

    if drift:
        _ui_alerts.append({'kind': 'drift', 'query': query, 'region': region_name, 'detail': drift})
        logger.warning(f'Stage 1b UI drift on "{query}" / {region_name}: {drift}')

    if filter_problems:
        _ui_alerts.append({'kind': 'filters', 'query': query, 'region': region_name,
                           'detail': '; '.join(filter_problems)})
        logger.warning(f'Stage 1b filters not applied on "{query}" / {region_name}: {"; ".join(filter_problems)}')
        return {'content': [{'type': 'text', 'text':
            'FILTERS NOT APPLIED: ' + '; '.join(filter_problems) + '. Re-apply the missing filter '
            'by clicking its chip, then call report_search again. If a chip is genuinely absent, '
            'say so and stop this search.'}]}

    # A filter caught missing and then re-clicked is the check working, not a problem: the
    # harvest can only follow this passing report. Only an alert never followed by one is real.
    for alert in _ui_alerts:
        if alert.get('kind') == 'filters' and alert.get('query') == query and alert.get('region') == region_name:
            alert['resolved'] = True

    return {'content': [{'type': 'text', 'text': 'ok — page looks sound, filters applied; harvest it'}]}


def region_overlap_report() -> dict[str, float]:
    """Jaccard overlap of harvested ids between regions, per query.

    A near-1.0 score means the location filter did nothing. On 2026-08-18 every query would have
    scored ~1.0 and nothing in the funnel could say so.
    """
    by_query: dict[str, list[set[str]]] = {}
    for (query, _region), ids in _search_ids.items():
        by_query.setdefault(query, []).append(ids)

    overlaps: dict[str, float] = {}
    for query, id_sets in by_query.items():
        if len(id_sets) < MIN_REGIONS_FOR_OVERLAP:
            continue
        worst = 0.0
        for i in range(len(id_sets)):
            for j in range(i + 1, len(id_sets)):
                union = id_sets[i] | id_sets[j]
                if not union:
                    continue
                worst = max(worst, len(id_sets[i] & id_sets[j]) / len(union))
        overlaps[query] = round(worst, REGION_OVERLAP_DECIMALS)
    return overlaps


def make_evaluator_server():
    """MCP server for stage 2: captures the condensed page extract from the Haiku extractor."""
    return create_sdk_mcp_server(
        name='job_evaluator', version='1.0.0',
        tools=[submit_job_extract],
    )
