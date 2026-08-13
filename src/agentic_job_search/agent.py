import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import re
import socket
import sys
import tempfile
import time
from datetime import date, datetime
from urllib.parse import quote_plus
import requests
import yaml
from pathlib import Path
from typing import Any

from agentic_job_search.config import (
    APPLIED_JOBS_HORIZON_DAYS,
    AUDIT_OPUS_SAMPLE_SIZE,
    EXTRACTOR_PROVIDER,
    JOB_STALE_AGE_DAYS,
    MAX_REFERENCE_JOBS,
    MAX_SEARCH_QUERIES,
    MODEL_NAME_HIGH,
    MODEL_NAME_LOW,
    MODEL_NAME_MEDIUM,
    QUERY_PROVIDER,
    RATING_PROVIDER,
    SCRAPER_DISALLOWED_BROWSER_TOOLS,
    SCRAPER_INTER_QUERY_DELAY_SECONDS,
    SCRAPER_INTER_SEARCH_DELAY_SECONDS,
    SCRAPER_MAX_LISTINGS_PER_SEARCH,
    SCRAPER_MAX_TURNS_PER_QUERY,
    SCRAPER_MIN_TURNS_PER_QUERY,
    SCRAPER_MIN_LISTINGS_PER_QUERY,
    REFERENCE_SUMMARY_MAX_CHARS,
    THINKING_MAX_CHARS,
    TRIAGE_ENABLED,
)
from agentic_job_search.extract_openrouter import extract_job_page_openrouter
import agentic_job_search.preferences as preferences
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
    extract_json_object,
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
# Playwright MCP snapshot/screenshot output — kept out of the repo tree
PLAYWRIGHT_OUTPUT_DIR = Path(tempfile.gettempdir()) / 'linkedin-agent-playwright-output'
TELEGRAM_MCP_URL = 'http://localhost:8004/mcp'
REFERENCE_SUMMARY_CACHE_PATH = RUN_DIR / 'reference_summary_cache.yaml'


async def _send_pipeline_notification(text: str) -> None:
    try:
        await call_mcp_tool(TELEGRAM_MCP_URL, 'send_message', {'text': text})
    except Exception as ex:
        console.print(f'[yellow]Warning: Telegram notification failed: {ex}[/yellow]')


# --- Prompt construction ---

AGENT_INSTRUCTIONS_COMMON_TEMPLATE = """You are a personal job search agent.

Do not write files directly to disk. Use the save_job_posting tool to persist job data — it is the only way you should store anything.

When browsing LinkedIn:
- Navigate to https://www.linkedin.com/jobs/ to search for jobs
- Extract: job title, company, location, remote/in-person/hybrid (if shown), salary (if shown), and key requirements
- **Before evaluating any job, call `check_and_record_job` with the site, job ID, posting date (YYYY-MM-DD), company, and job title. If it returns `already_processed`, `too_old`, `already_applied`, or `auth_required`, skip the job entirely. Only proceed with jobs that return `new`.**

The user's LinkedIn session is persisted so they should already be logged in. If not, ask them to log in via the browser.

{sponsorship_section}
## Rating jobs

Rate every job you evaluate on a 1–5 scale:
- 1 — Poor fit (missing key requirements or deal-breakers)
- 2 — Weak fit (some relevant aspects but significant gaps)
- 3 — Decent fit (meets most requirements, worth considering)
- 4 — Good fit (strong match on most criteria)
- 5 — Excellent fit (matches nearly everything)
"""

AGENT_INSTRUCTIONS_INTERACTIVE_TEMPLATE = AGENT_INSTRUCTIONS_COMMON_TEMPLATE + """
## Interactive mode

Your job is to:
1. Search LinkedIn for relevant job postings based on the user's resume and job requirements
2. Present job listings to the user one at a time and ask if they're interested
3. Learn from feedback (yes/no + reason) to refine future searches and build a picture of the ideal role
4. Eventually help draft cover letters and apply to approved jobs (email, web forms, or LinkedIn)

Present each job clearly, then ask the user if it's a good fit and why.
"""


def _sponsorship_prompt_section() -> str:
    """The visa-sponsorship hard rule, or '' when no sponsorship-required location is configured.

    Where the user needs sponsorship is personal (see preferences.py), so it is injected rather
    than written into the prompt text.
    """
    locations = preferences.sponsorship_required_in()
    if not locations:
        return ''
    joined = ', '.join(loc.title() for loc in locations)
    return (
        '\n## Hard rule — jobs requiring visa sponsorship\n'
        f'If the job is located in {joined} and the posting does NOT explicitly state that visa '
        'sponsorship is available (e.g., "we sponsor visas", "H-1B sponsorship available", '
        '"willing to sponsor"), rate it **1**.\n'
    )


def build_interactive_instructions() -> str:
    """Interactive-mode system prompt with the configured sponsorship rule injected."""
    return AGENT_INSTRUCTIONS_INTERACTIVE_TEMPLATE.format(
        sponsorship_section=_sponsorship_prompt_section()
    )

# Read-only harvest of every job card on a LinkedIn AI-powered results page.
#
# Kept out of SCRAPER_INSTRUCTIONS_TEMPLATE and injected as {harvest_js}: the JS is full of braces,
# which str.format() would read as placeholders. Raw string so the regex escapes survive.
#
# The job id is the `componentkey` suffix, which is why the scraper never has to click a card —
# and must not, since the only real <button> in a row is Dismiss. Cards appear twice in the DOM,
# hence the dedupe; each label is rendered twice (a visually-hidden copy carrying "(Verified job)"
# plus the visible one), hence `clean`.
SCRAPER_HARVEST_JS = r"""() => {
  const box = document.querySelector('div[componentkey="SearchResultsMainContent"]');
  if (!box) return {error: 'results container not found'};
  const seen = new Set(), out = [];
  const clean = el => {
    const parts = [...new Set([...el.childNodes]
      .map(n => (n.textContent || '').trim().replace(/\s+/g, ' ')).filter(Boolean))];
    return (parts.length ? parts[parts.length - 1] : (el.textContent || ''))
      .replace(/\s*\(Verified job\)\s*/i, '').replace(/\s+/g, ' ').trim();
  };
  for (const card of box.querySelectorAll('div[componentkey^="job-card-component-ref-"]')) {
    const id = (card.getAttribute('componentkey') || '').replace('job-card-component-ref-', '');
    if (!id || seen.has(id)) continue;
    seen.add(id);
    const ps = [...card.querySelectorAll('p')].map(clean).filter(Boolean);
    const meta = ps.slice(3);
    out.push({id, title: ps[0] || '', company: ps[1] || '', location: ps[2] || '',
      posted: (meta.find(t => /ago|Posted|Reposted/i.test(t)) || '')
        .replace(/^Posted\s*/i, '').replace(/(.+?)\1$/, '$1')});
  }
  return {count: out.length, jobs: out};
}"""

SCRAPER_INSTRUCTIONS_TEMPLATE = """You are a job listing scraper. Your job is to find new job postings on LinkedIn and add them to the internal evaluation queue.

You are fully authorized to call all available tools. Call them directly — do not ask for permission.

## MOST IMPORTANT RULE: you are driving a real person's real LinkedIn account

Every action you take happens in a logged-in session belonging to an actual job seeker. If that
account gets rate-limited, flagged, or banned, the entire job search ends — permanently, and with
consequences well beyond this run. **A slow, incomplete run is always better than an aggressive
one.** When in doubt, do less and pause longer.

This means, without exception:

- **NEVER click anything in the job results list.** Not a card, not a title, not a logo. The only
  real button in each row is **Dismiss**, and the accessibility tree disguises it as the card
  itself (see below). A single stray click permanently removes a job from the user's feed. You do
  not need to click: everything you need is read out of the DOM in Step 3.
- **Do not apply to anything, ever**, and do not save, follow, or dismiss.
- **Pause between searches.** Wait {search_delay_min}–{search_delay_max} seconds between one
  search and the next, using browser_wait_for. Vary it — never the same interval twice.
- **JavaScript may read the page, never drive it.** Use browser_evaluate to extract data and to
  scroll. Never write JavaScript that clicks, submits, or dispatches events.
- **Read before you act**, as a person would: take a snapshot, look at what is there, then act.
- **Stop immediately** if you see a CAPTCHA, "unusual activity", a verification challenge, or a
  forced re-login. Do not attempt to solve or work around it. Say clearly what you saw and stop
  the whole query. Being blocked once is recoverable; pushing through it is not.
- **Cap your effort**: at most {max_listings} listings per search. The tail of a results page is
  mostly already-seen or off-target, and is not worth the extra account exposure.

Take as long as you need and use as many words as you like reasoning about the page — verbosity
costs nothing here. Speed is the only thing that is expensive.

## How LinkedIn's job search works now (this changed on 2026-08-11)

This account is on LinkedIn's **AI-powered job search**. The page says so: *"You're now using
AI-powered job search. Some filters may no longer be available, but you can type them into search
to refine your results."* Two consequences drive everything below.

**1. Filters no longer exist as URL parameters or as chips.** `location`, `f_WT`, `f_E` and
`sortBy` are all stripped from the URL, and the old filter-chip row is gone. The search box is now
a natural-language field labelled "Describe the job you want". **Filters go into the query text**,
exactly as LinkedIn instructs — e.g. `Staff AI Engineer, remote, Canada, senior level`.

**2. The accessibility tree is a trap on this page — read job ids from the DOM instead.** Result
cards have no `<a href>`, and the only real `<button>` in each row is its **Dismiss** control,
whose `aria-label` is "Dismiss <job title> job". Because accessible names concatenate row text,
each card appears in a snapshot as one button labelled like
`button "Staff ML Engineer ... Samsara ... Dismiss ... job"`. **Clicking that ref clicks Dismiss**
— it removes the job from the user's feed and teaches LinkedIn to stop recommending similar roles.
An earlier version of this scraper destroyed three real jobs that way in under two minutes.

**So: never click anything in the results list.** You do not need to. The job id is already in
the DOM.

### Step 1 — open the results page

Navigate to:

   https://www.linkedin.com/jobs/search-results/?keywords=<QUERY+TEXT>&f_SAL=

`keywords` is the only parameter still honoured. The empty `f_SAL=` clears a leftover salary
filter that persists between sessions and otherwise silently narrows every search.

### Step 2 — confirm what you are actually looking at, and say so

Take a snapshot and state in your reply: the result count and the query text the search box
actually contains (LinkedIn rewrites it). If you see a sign-in wall, a challenge, or zero results,
say that explicitly and stop.

This read-back is the record of whether the search was real. Do not skip it.

### Step 3 — harvest every listing in ONE read-only call

The results container is `div[componentkey="SearchResultsMainContent"]`, and each job card is a
`div[componentkey="job-card-component-ref-<jobId>"]` inside it — **the job id is the attribute
suffix**. Scroll the results list to load the cards, then run exactly this with browser_evaluate:

```js
{harvest_js}
```

Cards are duplicated in the DOM, so the `seen` dedupe matters. This is the one place JavaScript is
correct here: it *reads* the page, it does not drive it.

### Step 4 — record what you harvested

State how many jobs came back. Then for each, up to {max_listings}:

1. Call **check_and_record_job** with site="linkedin", the harvested `id`, `company` and `title`.
   Pass `date_posted` from `posted` when present (e.g. "4 days ago"); omit it otherwise.
2. If it returns **"new"**, call **queue_candidate** with the URL
   `https://www.linkedin.com/jobs/view/<id>/`, plus title, company, the `location` as the snippet,
   date_posted if known, and query set to the search query string currently being processed.
3. If it returns "already_processed", "too_old", "already_applied", or "auth_required", move on.

If the harvest returns `{{error: ...}}` or zero jobs while the page visibly shows results, LinkedIn
has changed the markup. **Say so explicitly** — do not fall back to clicking the list. Do not
navigate to individual job pages and do not apply to anything; a separate evaluation agent visits
the job pages later.

## Search coverage

You are given ONE search query per session. Run it as {search_count} searches — one per target
region below — then stop. Do not invent additional queries:

{search_list}

Expect many `already_processed` results, especially on later searches. **That is expected and
correct — it is not a failure, and not a reason to skip the rest of a search.** The few new jobs
that come back are usually the best-matching ones you will find.

**But if two searches return the IDENTICAL list of jobs, the region text did not take effect.**
That is a failure, not a sign the query is exhausted. Say so explicitly rather than stopping
early.

If a search returns zero results, wait, then re-run it once before concluding there are none.

Only pass information directly visible on the card. Do not infer or fabricate missing fields —
Stage 2 will fill in the details from the job page.
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
- workplace_type: exactly one of "remote", "hybrid", or "onsite", whenever the page states the work arrangement. Any mention of required days in the office (e.g. "2-3 days onsite", "3 days per week in our Amsterdam office") is "hybrid", NOT "remote" — even when the search result or the header badge said Remote. Leave empty only if the page genuinely does not say.
- language_requirement: languages the posting explicitly REQUIRES (not nice-to-haves), comma-separated lowercase, e.g. "english, german". Leave empty if no language requirement is stated.
- relocation: if the posting requires the candidate to relocate to or reside in a specific country/city (e.g. "must be based in Portugal", "remote within Spain", "relocation to Madrid"), give that location. Leave empty for work-from-anywhere roles.
- education_requirement: "master" or "phd" ONLY if the posting states an advanced degree as a hard requirement (e.g. "MSc in Computer Science required", "PhD is a must"). Leave empty when the degree is merely preferred, when equivalent experience is accepted ("Master's or equivalent practical experience", "MSc a plus", "Bachelor's or Master's"), or when only a Bachelor's is required.

Do not rate the job. Do not browse other pages. Extract this one posting, submit it, then stop.
"""

EVALUATOR_INSTRUCTIONS_TEMPLATE = """You are evaluating a single job posting. A condensed extract of the posting is provided in the user message — you do not need to browse anywhere.
{sponsorship_section}{relocation_section}
## Workplace type
Check the `Workplace:` and `Location:` lines. Any requirement to be in an office some days a week is **hybrid**, even if the listing is badged "Remote".

- **Remote** — the expectation. No penalty.
- **Hybrid or on-site** — always a negative. Rate **4 or 5 only if BOTH**: (a) the location is one of the acceptable hybrid locations listed below; **and** (b) the role is strong in other respects, notably compensation well above target. If either fails, rate **3 at most**.
{hybrid_locations_section}
Rate the job 1–5 based on the requirements below:
- 1 — Poor fit (missing key requirements or deal-breakers)
- 2 — Weak fit (some relevant aspects but significant gaps)
- 3 — Decent fit (meets most requirements, worth considering)
- 4 — Good fit (strong match on most criteria)
- 5 — Excellent fit (matches nearly everything)

Also produce:
- reasoning: 2–3 sentences on the fit
- summary: a short label summarising the job (used in the saved filename)
- pros: 2–4 short bullet phrases (~100 chars each) naming the concrete strengths — matching tech, seniority, compensation, domain
- warnings: 0–4 short bullet phrases naming anything that conflicts with the requirements above — hybrid/on-site, contract vs full-time, salary below target, missing salary, stack mismatch, language expectations. Every conflict you notice MUST appear here, even when you still rate the job highly.

Do NOT write a warning about the poster being a recruiting agency or the hiring company being undisclosed — that is detected deterministically and added for you, and repeating it just duplicates the bullet in different words. Being posted by an agency is **not** a reason to lower the rating; judge the role itself.
"""


def build_scraper_instructions() -> str:
    """Scraper prompt with the configured search regions injected.

    Regions are a personal preference (see preferences.py), so the search list is generated
    rather than hardcoded. With no regions configured, the query runs unfiltered in both
    sort orders.

    Regions become *query text*, not URL parameters or filter chips. LinkedIn's AI-powered job
    search strips `location`/`f_WT`/`f_E`/`sortBy` from the URL and has no filter-chip row; its
    own guidance is to type filters into the search box (see the 2026-08-11 entry in CLAUDE.md).

    Sort order is no longer a coverage axis — the AI-powered UI exposes no sort control, so the
    former "each region in both sort orders" fan-out would just be the same search run twice.
    Region is the one axis that still varies the result set.
    """
    regions = preferences.search_regions()
    entries: list[str] = []
    if regions:
        for region in regions:
            name = region.get('name') or region.get('linkedin_location', '')
            location = str(region.get('linkedin_location', ''))
            entries.append(f'**{name}** — search text: `<QUERY>, remote, {location}, senior level`')
    else:
        entries.append('**Unfiltered** — search text: `<QUERY>, remote, senior level`')

    search_list = '\n'.join(f'{i}. {entry}' for i, entry in enumerate(entries, start=1))
    search_min, search_max = SCRAPER_INTER_SEARCH_DELAY_SECONDS
    return SCRAPER_INSTRUCTIONS_TEMPLATE.format(
        search_count=len(entries),
        search_list=search_list,
        search_delay_min=search_min,
        search_delay_max=search_max,
        max_listings=SCRAPER_MAX_LISTINGS_PER_SEARCH,
        harvest_js=SCRAPER_HARVEST_JS,
    )


def build_evaluator_instructions() -> str:
    """Evaluator prompt with the configured sponsorship, relocation, and hybrid rules injected."""
    sponsorship_section = _sponsorship_prompt_section()
    note = preferences.relocation_note()
    relocation_section = f'\n## Relocation (applies to REMOTE roles only)\n{note}\n' if note else ''

    locations = preferences.hybrid_acceptable_locations()
    if locations:
        joined = ', '.join(loc.title() for loc in locations)
        hybrid_locations_section = (
            f'- Acceptable hybrid/on-site locations: {joined}.\n'
            '- Hybrid or on-site anywhere else is **not a fit** however well the role itself '
            'matches. Never rate those 4 or 5.\n'
        )
    else:
        hybrid_locations_section = (
            '- No hybrid/on-site location is acceptable: rate every hybrid or on-site role '
            '**3 at most**.\n'
        )

    return EVALUATOR_INSTRUCTIONS_TEMPLATE.format(
        sponsorship_section=sponsorship_section,
        relocation_section=relocation_section,
        hybrid_locations_section=hybrid_locations_section,
    )

EXTRACT_OUTPUT_SCHEMA = {
    'type': 'object',
    'properties': {
        'title': {'type': 'string'},
        'company': {'type': 'string'},
        'description': {'type': 'string', 'description': 'Condensed posting content (requirements, responsibilities, stack, seniority)'},
        'location': {'type': 'string', 'description': 'Location including workplace type (Remote / Hybrid / On-site)'},
        'workplace_type': {'type': 'string', 'description': "Work arrangement: 'remote', 'hybrid', or 'onsite'"},
        'date_posted': {'type': 'string', 'description': 'As shown on the page, absolute or relative'},
        'closed': {'type': 'boolean', 'description': 'True if the page shows "No longer accepting applications"'},
        'salary': {'type': 'string'},
        'sponsorship_note': {'type': 'string', 'description': 'Any visa/work-authorization statement, verbatim'},
        'language_requirement': {'type': 'string', 'description': "Explicitly required languages, comma-separated lowercase, e.g. 'english, german'"},
        'relocation': {'type': 'string', 'description': 'Location the candidate must relocate to / reside in, if the posting requires one'},
        'education_requirement': {'type': 'string', 'description': "'master' or 'phd' ONLY if an advanced degree is a HARD requirement (e.g. 'MSc required', 'PhD is a must'); empty when merely preferred, when equivalent experience is accepted, or when only a Bachelor's is required"},
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
        'pros': {
            'type': 'array', 'items': {'type': 'string'},
            'description': '2-4 short bullet phrases naming concrete strengths of the role',
        },
        'warnings': {
            'type': 'array', 'items': {'type': 'string'},
            'description': '0-4 short bullet phrases naming anything conflicting with the requirements',
        },
    },
    'required': ['rating', 'company', 'title', 'reasoning', 'summary', 'pros', 'warnings'],
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

    parts.append(build_interactive_instructions())

    return "\n\n".join(parts)


def build_scraper_prompt() -> str:
    # No date needed — check_and_record_job enforces age filtering via tool.
    parts = []
    if JOB_REQUIREMENTS_PATH.exists():
        requirements = JOB_REQUIREMENTS_PATH.read_text(encoding="utf-8")
        parts.append(f"--- JOB_REQUIREMENTS.md ---\n{requirements}\n--- END JOB_REQUIREMENTS.md ---")
    parts.append(build_scraper_instructions())
    return "\n\n".join(parts)


def build_query_generation_instructions() -> str:
    """Query-generation prompt with the configured title preferences injected."""
    lines = [
        'Generate short LinkedIn job search queries (2–6 words each, like a job title).',
        f'Return AT MOST {MAX_SEARCH_QUERIES} queries — each one costs two live LinkedIn searches, '
        'so they must be the highest-yield titles, not an exhaustive list of variations.',
        'The jobs I have applied to are the strongest signal of what I want: cover the title space '
        'they occupy, and include adjacent titles likely to surface similar roles I have not seen yet.',
    ]
    preferred = preferences.preferred_titles_note()
    if preferred:
        lines.append(preferred)
    excluded = preferences.excluded_title_words()
    if excluded:
        joined = ', '.join(f'"{word}"' for word in excluded)
        lines.append(
            f'Do NOT generate titles containing any of these — {joined}, or similar; those are not '
            'the roles I want.'
        )
    lines.append(
        'Prefer broad, common titles that LinkedIn actually returns results for over narrow or '
        'invented ones. Queries are job titles, not company names — never search for a company.'
    )
    return '\n'.join(lines)

QUERY_JSON_INSTRUCTIONS = (
    'Respond with ONLY a JSON object: {"queries": ["<query>", "<query>", ...]}'
)


async def _generate_queries_openrouter(prompt: str, stage_stats: dict | None) -> list[str]:
    """Query generation via the OpenRouter MCP server (glm). Raises on any failure."""
    content, cost_usd = await chat_openrouter(f'{prompt}\n\n{QUERY_JSON_INSTRUCTIONS}')
    if stage_stats is not None:
        stage_stats['cost'] += cost_usd
    queries = extract_json_object(content).get('queries') or []
    if not isinstance(queries, list) or not queries:
        raise ValueError(f'OpenRouter returned no usable queries: {content[:300]!r}')
    return [str(q).strip() for q in queries if str(q).strip()]


async def generate_search_queries(stage_stats: dict | None = None) -> list[str]:
    """Derive LinkedIn search queries from resume + requirements + applied jobs.

    Provider chain: OpenRouter (glm) -> Anthropic tool-call. Capped at MAX_SEARCH_QUERIES.
    """
    parts = []
    resume = load_resume()
    if resume:
        parts.append(f'--- RESUME ---\n{resume}\n--- END RESUME ---')
    if JOB_REQUIREMENTS_PATH.exists():
        requirements = JOB_REQUIREMENTS_PATH.read_text(encoding='utf-8')
        parts.append(f'--- JOB_REQUIREMENTS.md ---\n{requirements}\n--- END JOB_REQUIREMENTS.md ---')
    applied = tools_module.applied_jobs_summary()
    if applied:
        parts.append(
            f'--- JOBS I HAVE APPLIED TO (last {APPLIED_JOBS_HORIZON_DAYS} days) ---\n{applied}\n'
            '--- END JOBS I HAVE APPLIED TO ---'
        )
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
                    'description': f'At most {MAX_SEARCH_QUERIES} LinkedIn search queries of 2–6 words each, e.g. ["Staff ML Engineer", "Principal AI Engineer"]',
                },
            },
            'required': ['queries'],
        },
    )
    async def _submit(args: dict[str, Any]) -> dict:
        captured.extend(args['queries'])
        return {'content': [{'type': 'text', 'text': 'Queries submitted.'}]}

    query_server = create_sdk_mcp_server(name='query_generator', version='1.0.0', tools=[_submit])

    prompt = f'{context}\n\n{build_query_generation_instructions()}'

    provider = ''
    if QUERY_PROVIDER == 'openrouter':
        try:
            captured = await _generate_queries_openrouter(prompt, stage_stats)
            provider = 'openrouter'
        except Exception as ex:
            logger.warning(f'Stage 1a via OpenRouter failed, falling back to Anthropic: {ex}')

    if not captured:
        options = ClaudeAgentOptions(
            tools=[],
            model=MODEL_NAME_MEDIUM,
            system_prompt='You are a tool-calling assistant. Always respond by calling the provided tool — never respond with text.',
            mcp_servers={'query_generator': query_server},
            allowed_tools=['mcp__query_generator__submit_search_queries'],
            permission_mode='bypassPermissions',
            cwd=str(PROJECT_DIR),
        )
        try:
            async for msg in sdk_query(prompt=f'{prompt}\n\nCall the submit_search_queries tool with your list.', options=options):
                if isinstance(msg, ResultMessage):
                    cost_delta = accumulate_stage_stats(stage_stats, msg) if stage_stats is not None else None
                    print_result_stats(msg, cost_delta)
            provider = 'anthropic'
        except Exception as ex:
            raise RuntimeError(f'Stage 1a (query generation) failed: {ex}') from ex

    if not captured:
        raise ValueError('Query generation produced no queries on any provider')

    if len(captured) > MAX_SEARCH_QUERIES:
        logger.info(f'Stage 1a: trimming {len(captured)} queries to the {MAX_SEARCH_QUERIES} cap: dropped {captured[MAX_SEARCH_QUERIES:]}')
        captured = captured[:MAX_SEARCH_QUERIES]
    logger.info(f'Stage 1a: generated {len(captured)} search queries via {provider}: {captured}')
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
    parts.append(build_evaluator_instructions())
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


def log_agent_text(stage: str, text: str) -> None:
    """Mirror an agent TextBlock to the run log.

    The agent loops print() their text blocks, which reaches the terminal only. That is where
    the scraper reports what it actually saw — 'these two searches returned identical
    results', 'the Remote filter did not stick', 'sign-in wall'. On 2026-08-11 a run collapsed
    from 166 listings to 7 and the model's own account of why was discarded, leaving only
    counters to diagnose from. Thinking blocks stay console-only; they are long and the
    findings live in the text blocks.
    """
    for line in text.strip().splitlines():
        if line.strip():
            logger.info(f'{stage}: {line.strip()}')


def print_result_stats(msg: ResultMessage, cost_delta: float | None = None) -> None:
    parts = []
    if msg.usage:
        parts.append(f"in={msg.usage.get('input_tokens', 0)} out={msg.usage.get('output_tokens', 0)}")
        cache_read = msg.usage.get("cache_read_input_tokens", 0)
        cache_write = msg.usage.get("cache_creation_input_tokens", 0)
        if cache_read or cache_write:
            parts.append(f"cache_read={cache_read} cache_write={cache_write}")
    if msg.total_cost_usd is not None:
        parts.append(f"cost=${msg.total_cost_usd:.4f}")
        if cost_delta is not None and abs(cost_delta - msg.total_cost_usd) > 1e-9:
            parts.append(f"delta=${cost_delta:.4f}")
    if parts:
        console.print(f"[dim]{' · '.join(parts)}[/dim]")
    # The console output above is rich-only and never reaches run_dir/logs/run-*.log, which is
    # why a 13x scraping cost regression went unnoticed for nine runs. Mirror it to the log,
    # with the session identity and turn count needed to tell a cumulative cost field from a
    # per-request one.
    usage = msg.usage or {}
    logger.info(
        f'Result stats: session={msg.session_id} turns={msg.num_turns} '
        f'in={usage.get("input_tokens", 0)} out={usage.get("output_tokens", 0)} '
        f'cache_read={usage.get("cache_read_input_tokens", 0)} '
        f'cache_write={usage.get("cache_creation_input_tokens", 0)} '
        f'cost_reported={msg.total_cost_usd} '
        f'cost_charged={cost_delta if cost_delta is not None else msg.total_cost_usd}'
    )


# Keys that are internal bookkeeping, not reported metrics. stage_stats dicts are serialized
# straight into cost_log.jsonl, so these are stripped before logging to keep its schema stable.
_PRIVATE_STAGE_STAT_KEYS = ('session_costs',)


def new_stage_stats() -> dict:
    return {
        "cost": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        # session_id -> highest total_cost_usd seen for that session (see accumulate_stage_stats)
        "session_costs": {},
    }


def public_stage_stats(stage_stats: dict) -> dict:
    """Strip internal bookkeeping keys so cost_log.jsonl keeps its existing stage schema."""
    return {
        stage: {k: v for k, v in stats.items() if k not in _PRIVATE_STAGE_STAT_KEYS}
        for stage, stats in stage_stats.items()
    }


def accumulate_stage_stats(stats: dict, msg: ResultMessage) -> float:
    """Fold a ResultMessage into a stage's totals. Returns the cost actually charged."""
    if msg.usage:
        stats["input_tokens"] += msg.usage.get("input_tokens", 0)
        stats["output_tokens"] += msg.usage.get("output_tokens", 0)
        stats["cache_read_input_tokens"] += msg.usage.get("cache_read_input_tokens", 0)
        stats["cache_creation_input_tokens"] += msg.usage.get("cache_creation_input_tokens", 0)
    delta = cost_delta_for(stats, msg)
    stats["cost"] += delta
    return delta


def cost_delta_for(stats: dict, msg: ResultMessage) -> float:
    """Charge for a ResultMessage, recording it against its session.

    ``total_cost_usd`` is cumulative for the life of a session, so a stage that issues several
    requests on one shared ClaudeSDKClient (Stage 1b sends one request per search query) would
    be billed 1+2+...+N times over by naive summation. Charging only the increase over what
    that session has already reported bills each session exactly once.

    The bookkeeping is per session_id rather than a single running high-water mark because
    stages differ: Stage 1b shares one session across queries, while extraction opens a fresh
    session per job whose cumulative totals are unrelated to the previous job's.
    """
    if msg.total_cost_usd is None:
        return 0.0
    # No session id => nothing to correlate against, so treat the value as already per-request.
    if not msg.session_id:
        return msg.total_cost_usd
    session_costs = stats.setdefault("session_costs", {})
    previous = session_costs.get(msg.session_id, 0.0)
    delta = msg.total_cost_usd - previous
    if delta < 0:
        logger.warning(
            f'Session {msg.session_id} reported a lower cumulative cost '
            f'({msg.total_cost_usd}) than previously seen ({previous}); charging 0 for this '
            'request. total_cost_usd was expected to be non-decreasing within a session.'
        )
        delta = 0.0
    session_costs[msg.session_id] = max(msg.total_cost_usd, previous)
    return delta


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
                        log_agent_text('Interactive', block.text)
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


def _listings_seen() -> int:
    """Total job listings the scraper has inspected this run (any check_and_record_job outcome)."""
    return sum(tools_module._check_status_counts.values())


async def _human_pause(delay_range: tuple[float, float], reason: str) -> None:
    """Sleep a randomised interval to keep the scraper's access pattern human-shaped.

    Enforced in CODE, not in the prompt: within-query pacing depends on the model choosing to
    call browser_wait_for, and a model under turn pressure will skip it. Between queries there is
    no such ambiguity, so this is the one pause that always happens. The interval is drawn
    uniformly because a fixed delay is itself a robotic signature.
    """
    seconds = random.uniform(*delay_range)
    logger.info(f'Pausing {seconds:.1f}s before {reason} (human-emulation pacing)')
    await asyncio.sleep(seconds)


def _check_status_snapshot() -> dict[str, int]:
    """Copy of the global check_status counters, for before/after per-query diffing."""
    return dict(tools_module._check_status_counts)


def _check_status_delta(before: dict[str, int]) -> dict[str, int]:
    """Per-status counts accrued since `before`; zero-delta statuses are omitted."""
    return {
        status: count - before.get(status, 0)
        for status, count in tools_module._check_status_counts.items()
        if count - before.get(status, 0) > 0
    }


async def run_scraper(client: ClaudeSDKClient, queries: list[str], stage_stats: dict) -> None:
    """Stage 1: haiku scraper collects candidates from LinkedIn search results.

    ONE REQUEST PER QUERY, on a SINGLE shared session. Two constraints have to hold at once:

    - max_turns applies per request, not per session. Sending every query in one request
      lets the first few exhaust the budget (each check_and_record_job / queue_candidate
      call burns a turn) so the rest are never searched at all — while still being reported
      as "0 jobs", indistinguishable from "searched and found nothing". One request per
      query gives each its own turn budget.
    - The Playwright browser must not be re-attached per query. Opening a fresh
      ClaudeSDKClient for each query makes later sessions fail with "browser is in use"
      against the shared browser context, which looks exactly like an auth wall in the logs.

    Discovery resilience: if a query inspects fewer than SCRAPER_MIN_LISTINGS_PER_QUERY
    listings, retry it once. The two failure shapes need opposite corrections:

    - ZERO listings is the signature of an auth wall, block page, or empty results shell.
      Retry with the region text dropped.
    - A HANDFUL of listings means the page loaded but the listings were never walked. On
      2026-08-11 every query returned exactly one listing: LinkedIn's AI-powered results list
      exposes no job ids, so only the preselected card was identifiable, and the model stopped
      after 10 of 90 turns. A `seen == 0` trigger sails straight past that.

    Pacing: queries are separated by a randomised pause. This is a real logged-in account, and
    a flagged or banned account ends the whole job search — so pacing is enforced here in code
    rather than left to the model, which under turn pressure will skip a prompted wait.
    """
    async def run_pass(instruction: str) -> int | None:
        """Run one scraper request; returns the turn count the SDK reported, if any."""
        num_turns: int | None = None
        await client.query(instruction)
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, ThinkingBlock):
                        print_thinking(block.thinking)
                    elif isinstance(block, TextBlock):
                        print(block.text, flush=True)
                        log_agent_text('Stage 1b', block.text)
            elif isinstance(msg, ResultMessage):
                cost_delta = accumulate_stage_stats(stage_stats, msg)
                print_result_stats(msg, cost_delta)
                num_turns = msg.num_turns
        return num_turns

    for i, query in enumerate(queries, 1):
        console.print(f"[cyan]Stage 1b: query {i}/{len(queries)} — \"{query}\"[/cyan]")
        if i > 1:
            await _human_pause(SCRAPER_INTER_QUERY_DELAY_SECONDS, f'query {i}/{len(queries)}')
        tools_module._current_query = query
        before = _check_status_snapshot()
        try:
            turns = await run_pass(
                f'Search LinkedIn for this ONE query only: "{query}"\n\n'
                "Run every search listed in your instructions (one per region), putting the region "
                "and the filter words into the search text. Report the result count and the query "
                "text the search box actually contains. Then harvest the whole results list with "
                "the single read-only browser_evaluate from your instructions, and call "
                "check_and_record_job (and queue_candidate for new jobs) for each harvested "
                "listing. Never click anything in the results list — the card and its Dismiss "
                "button are indistinguishable to you, and a stray click destroys a real job. Do "
                "not ask for permission — call the tools directly. Do not navigate to individual "
                "job pages. Stop when you have worked through the searches for this query."
            )
        except Exception as ex:
            # One failed query must not abort the remaining ones.
            logger.warning(f'Stage 1b: query "{query}" failed: {ex}')
            tools_module._queries_searched[query] = 'error'
            continue

        delta = _check_status_delta(before)
        seen = sum(delta.values())
        if seen < SCRAPER_MIN_LISTINGS_PER_QUERY:
            if seen == 0:
                logger.warning(
                    f'Stage 1b: query "{query}" inspected 0 listings — retrying once with the '
                    'region text dropped (possible auth wall, block page, or empty results shell).'
                )
                retry_instruction = (
                    f'That search surfaced no job listings at all for "{query}" — the results list was '
                    'empty or you hit a sign-in / verification wall.\n\n'
                    'If it was a CAPTCHA, a verification challenge, or an "unusual activity" notice: do '
                    'NOT retry. Say what you saw and stop — pushing through a block is the one thing '
                    'that can end this account.\n\n'
                    'Otherwise retry once: load '
                    f'https://www.linkedin.com/jobs/search-results/?keywords={quote_plus(query)}&f_SAL= '
                    'with NO region words in the search text (keep "remote" and "senior level"), state '
                    'the result count you actually see, then walk the listings as instructed. If you '
                    'STILL see a wall or genuinely zero results, say so explicitly and stop.'
                )
            else:
                logger.warning(
                    f'Stage 1b: query "{query}" inspected only {seen} listing(s) '
                    f'(below {SCRAPER_MIN_LISTINGS_PER_QUERY}) — retrying once; the results list was '
                    'likely never harvested.'
                )
                retry_instruction = (
                    f'The query "{query}" inspected only {seen} job listing(s). A real search returns '
                    'roughly 25 on the first page, so the results list was not actually harvested.\n\n'
                    'Retry now: scroll the results list to load the cards, then run the single '
                    'read-only browser_evaluate harvest from your instructions — the one reading '
                    'div[componentkey^="job-card-component-ref-"] inside '
                    'div[componentkey="SearchResultsMainContent"], where the job id is the attribute '
                    'suffix. Report how many jobs it returned, then call check_and_record_job for '
                    'EVERY one, including those you expect to be already processed.\n\n'
                    'Do NOT click anything in the results list to work around this. The card and its '
                    'Dismiss button are the same element to you, and clicking destroys a real job. If '
                    'the harvest returns an error or zero jobs while results are visible on screen, '
                    'the markup has changed — say so and stop.\n\n'
                    'Stop immediately if a challenge or verification page appears. If two searches '
                    'return the identical list of jobs, the region text is not applying — say so '
                    'explicitly rather than treating the query as exhausted.'
                )
            await _human_pause(SCRAPER_INTER_SEARCH_DELAY_SECONDS, 'the recovery pass')
            try:
                await run_pass(retry_instruction)
            except Exception as ex:
                logger.warning(f'Stage 1b: recovery pass for "{query}" failed: {ex}')
            delta = _check_status_delta(before)
            seen = sum(delta.values())

        tools_module._queries_searched[query] = seen
        tools_module._check_status_per_query[query] = delta
        # Too few turns means the scraper bailed before completing even one harvest cycle
        # (navigate, snapshot, scroll, evaluate, then a record call per listing).
        #
        # This is an ABSOLUTE floor, not a fraction of the budget. A fraction was wrong: the
        # budget is sized for the worst case, so `budget // 3` fired on every healthy query once
        # harvesting replaced click-to-reveal and a full query legitimately took ~50 of 180 turns.
        # A warning that fires on success trains the reader to ignore it.
        if turns is not None and turns < SCRAPER_MIN_TURNS_PER_QUERY:
            logger.warning(
                f'Stage 1b: query "{query}" used only {turns} turns on its first pass (expected at '
                f'least {SCRAPER_MIN_TURNS_PER_QUERY}) — it stopped before completing a harvest '
                'cycle, which usually means it treated repeated or empty results as "done".'
            )
        logger.info(
            f'Stage 1b: query "{query}" inspected {seen} listing(s) '
            f'[{tools_module.format_status_counts(delta)}], '
            f'{tools_module._candidates_per_query.get(query, 0)} queued'
        )

    logger.info(
        f'Stage 1b complete: {len(queries)} quer(ies) searched, {_listings_seen()} listing(s) inspected, '
        f'{len(tools_module._candidates)} candidate(s) queued'
    )


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
                        log_agent_text('Stage 2a', block.text)
            elif isinstance(msg, ResultMessage):
                cost_delta = accumulate_stage_stats(stage_stats, msg)
                print_result_stats(msg, cost_delta)
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
            cost_delta = accumulate_stage_stats(stage_stats, msg)
            print_result_stats(msg, cost_delta)
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
        'workplace_type': (structured.get('workplace_type') or '').strip().lower(),
        'education_requirement': (structured.get('education_requirement') or '').strip().lower(),
    }
    logger.info(
        f"Extract fallback: {candidate['company']} — {candidate['title']}: "
        f"{len(snapshot)} snapshot chars → {len(extract['description'])} chars condensed"
    )
    return extract


_ONSITE_RE = re.compile(r'\bon[\s-]?site\b|\bin[\s-]?office\b|\bin[\s-]?person\b', re.IGNORECASE)
# "N days" wording is the tell for hybrid and must beat the on-site check: a posting saying
# "on site 3 days per week" is hybrid, and one saying "2-3 days onsite" often also carries a
# "Remote" badge from the job board's own filter.
_HYBRID_RE = re.compile(
    r'\bhybrid\b'
    r'|\b\d\s*(?:-\s*\d\s*)?days?\s+(?:a|per)\s+week\b'
    r'|\b\d\s*(?:-\s*\d\s*)?days?\s+(?:in|at|on)[\s-]?(?:the\s+|our\s+)?(?:office|site)\b'
    r'|\b\d\s*(?:-\s*\d\s*)?days?\s+on[\s-]?site\b',
    re.IGNORECASE,
)
_REMOTE_RE = re.compile(r'\b(?:fully\s+)?remote\b|\bwork\s+from\s+(?:home|anywhere)\b', re.IGNORECASE)


def derive_workplace_type(extract: dict) -> str:
    """'remote' | 'hybrid' | 'onsite' | '' — the extractor's structured value when it set one,
    otherwise inferred from the free-text location/description.

    The inference matters: workplace_type is new and the extractors do not always fill it, but the
    workplace wording has always leaked into `location` (e.g. 'Netherlands (Hybrid - 2-3 days
    onsite)'). Hybrid is checked before remote — postings routinely say both, and a posting that
    mentions any required office days is hybrid regardless of a 'Remote' badge.
    """
    explicit = str(extract.get('workplace_type') or '').strip().lower()
    if explicit in {'remote', 'hybrid', 'onsite'}:
        return explicit
    if explicit in {'on-site', 'on site', 'in-office', 'in office'}:
        return 'onsite'

    haystack = f"{extract.get('location', '')}\n{extract.get('description', '')[:2000]}"
    if _HYBRID_RE.search(haystack):
        return 'hybrid'
    if _ONSITE_RE.search(haystack):
        return 'onsite'
    if _REMOTE_RE.search(haystack):
        return 'remote'
    return ''


_PHD_RE = re.compile(r"\bph\.?\s?d\.?\b|\bdoctorates?\b|\bdoctoral\s+degree\b", re.IGNORECASE)
_MASTERS_RE = re.compile(
    r"\bmasters?'?s?\s+degree\b|\bmaster's\b|\bmasters\b|\bm\.?sc\.?\b"
    r"|\bmaster\s+of\s+(?:science|engineering|arts)\b|\bm\.?s\.?\s+in\b|\bm\.?a\.?\s+in\b",
    re.IGNORECASE,
)
_DEGREE_REQUIRED_RE = re.compile(
    r"\brequired\b|\brequirements?\b|\brequires?\b|\bmust\s+(?:have|hold|possess)\b"
    r"|\bis\s+a\s+must\b|\bmandatory\b|\bminimum\b",
    re.IGNORECASE,
)
# Any of these in the same sentence means the degree is not a hard gate: it is preferred, one of
# several accepted options, or substitutable with experience. A bachelor's mention counts as a
# softener because an advanced degree cannot be mandatory if a bachelor's is also acceptable.
_DEGREE_SOFTENER_RE = re.compile(
    r"\bpreferr?ed\b|\bpreferabl[ey]\b|\bor\s+equivalent\b|\bequivalent\s+(?:practical\s+)?experience\b"
    r"|\bnice\s+to\s+have\b|\ba\s+plus\b|\bbonus\b|\bideally\b|\bdesirable\b|\badvantage\b"
    r"|\bbachelor'?s?\b|\bb\.?sc\.?\b",
    re.IGNORECASE,
)
# Negations are as disqualifying as softeners: postings routinely advertise "No PhD required" as a
# selling point, and the rater's own prose ("degrees are not required") lands in the same text.
_DEGREE_NEGATION_RE = re.compile(
    r"\bno\s+(?:\w+\s+){0,2}(?:degree|phd|ph\.?\s?d\.?|master'?s?|requirements?)\b"
    r"|\bnot\s+(?:strictly\s+)?(?:required|mandatory|a\s+requirement)\b"
    r"|\bwithout\s+(?:a\s+)?(?:phd|ph\.?\s?d\.?|master'?s?|degree)\b"
    r"|\bdegrees?\s+(?:are\s+)?not\b"
    r"|\bdo(?:es)?\s+not\s+require\b",
    re.IGNORECASE,
)
# Split on sentence-final periods only (followed by whitespace or end), so 'Ph.D.' survives intact.
_SENTENCE_SPLIT_RE = re.compile(r'[;•\n\r]|\.(?=\s|$)|(?<=[a-z])\s*[-–—]\s+')


def derive_education_requirement(extract: dict) -> str:
    """'master' | 'phd' | '' — the advanced degree the posting states as a HARD requirement.

    The extractor's structured value wins when it set one. Otherwise the description is scanned
    sentence by sentence, and a sentence only counts when it names a degree AND a requirement word
    AND carries no softener ('preferred', 'or equivalent', a bachelor's alternative) and no negation
    ('No PhD required'). Those two checks are the load-bearing part: an advanced-degree rule is an
    unappealable auto-reject, so a posting that would take experience instead must never match.
    """
    explicit = str(extract.get('education_requirement') or '').strip().lower()
    if explicit:
        if _PHD_RE.search(explicit):
            return 'phd'
        if _MASTERS_RE.search(explicit) or explicit in {'master', 'ms', 'ma'}:
            return 'master'

    haystack = f"{extract.get('location', '')}\n{extract.get('description', '')[:4000]}"
    for sentence in _SENTENCE_SPLIT_RE.split(haystack):
        if not _DEGREE_REQUIRED_RE.search(sentence):
            continue
        if _DEGREE_SOFTENER_RE.search(sentence) or _DEGREE_NEGATION_RE.search(sentence):
            continue
        if _MASTERS_RE.search(sentence):
            return 'master'
        if _PHD_RE.search(sentence):
            return 'phd'
    return ''


# Recruiter language in a job description. Deliberately PHRASES, not bare keywords: the historical
# corpus contains "AgencyAnalytics" (a real product company) and plenty of ordinary postings that
# mention "clients" in the work itself ("you will present to clients"). A phrase like "our client
# is" only shows up when someone is posting on another company's behalf.
_AGENCY_PHRASE_RE = re.compile(
    r'\b(on behalf of (?:our|a|the) client'
    r'|our client(?: is| are|,|\'s)'
    r'|my client(?: is| are|,)'
    r'|the client(?: is| are) (?:a|an|one)'
    r'|confidential (?:client|search)'
    r'|we are (?:recruiting|hiring) (?:for|on behalf of)'
    r'|recruiting on behalf of'
    r'|(?:our|a) (?:leading|major|well-known|prestigious) client)\b',
    re.IGNORECASE,
)

# Agency markers in the POSTER'S NAME. Anchored to word boundaries and to multi-word forms where a
# single word would over-match: "talent" alone hits "Talent.com" and product companies, and "search"
# alone hits half the AI industry.
_AGENCY_COMPANY_RE = re.compile(
    r'\b(recruit(?:ing|ment|ers?)?'
    r'|staffing'
    r'|talent (?:solutions|partners|acquisition|group|advisory)'
    r'|executive search'
    r'|headhunt(?:ing|ers?)'
    r'|manpower'
    r'|consultants?)\b',
    re.IGNORECASE,
)


def derive_end_client(extract: dict) -> str:
    """The hiring company behind a posting, '' when it is just the poster under another name.

    Extractors routinely echo the poster back as the end client for ordinary direct postings —
    measured on real jobs, glm returned end_client='lululemon' for a lululemon posting. Taken
    literally that would print a redundant "Hiring company: lululemon" beside "Company: lululemon"
    and spend an already-applied lookup re-checking a name Stage 1b already cleared. An end client
    is only interesting when it differs from who posted.
    """
    end_client = str(extract.get('end_client') or '').strip()
    if not end_client:
        return ''
    if tools_module._normalize_company(end_client) == tools_module._normalize_company(str(extract.get('company') or '')):
        return ''
    return end_client


def derive_agency_posting(extract: dict) -> bool:
    """True when the posting is by a staffing firm / recruiter / aggregator, not the employer.

    The extractor's structured `is_agency` wins whenever it answered at all — including a `False`,
    which is a judgement and not a gap. Only when it is None do we fall back to scanning, so a
    model that looked at the page and said "no" is never overridden by a regex.

    Conservative by design, but the cost of being wrong is asymmetric and small: this drives a
    WARNING only. It never rejects a job and never changes a rating (see the Account of decisions
    in CLAUDE.md) — so a false positive is a stray bullet in a notification, not a lost job.
    """
    explicit = extract.get('is_agency')
    if explicit is not None:
        return bool(explicit)
    if _AGENCY_COMPANY_RE.search(str(extract.get('company') or '')):
        return True
    return bool(_AGENCY_PHRASE_RE.search(str(extract.get('description') or '')[:4000]))


def format_extract_text(candidate: dict, extract: dict) -> str:
    lines = [
        f"Title: {extract['title']}",
        f"Company: {extract['company']}",
        f"Location: {extract['location']}",
        f"Posted: {extract['date_posted'] or candidate['date_posted']}",
        f"URL: {candidate['url']}",
    ]
    workplace_type = derive_workplace_type(extract)
    if workplace_type:
        lines.insert(3, f"Workplace: {workplace_type}")
    # Who would actually hire, when a recruiter names them — the poster's name is not the employer.
    if end_client := derive_end_client(extract):
        lines.insert(2, f"Hiring company: {end_client}")
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
    if extract.get('education_requirement'):
        lines.append(f"Education requirement: {extract['education_requirement']}")
    if extract['closed']:
        lines.append("Status: no longer accepting applications")
    lines.append(f"\n{extract['description']}")
    return '\n'.join(lines)


async def apply_hard_rules(candidate: dict, extract: dict) -> str | None:
    """Auto-reject rules; returns the reason, or None if the job survives.

    All rules but the first are deterministic and free. The blacklist runs first because it is
    the most decisive and short-circuits for free when the company name does not match; only an
    exact name hit costs a (cheap) confirmation call. See company_blacklist_reason().
    """
    full_text = format_extract_text(candidate, extract)

    # Both the poster and the end client: a recruiting agency can repost a blacklisted company's
    # role under its own name, which is the same gap end-client dedup closes for applied jobs.
    blacklist_context = '\n'.join(filter(None, [
        f"Company: {extract.get('company') or candidate.get('company', '')}",
        f"Location: {extract.get('location') or ''}",
        f"Title: {extract.get('title') or candidate.get('title', '')}",
        (extract.get('description') or '')[:500],
    ]))
    for name in (extract.get('company') or candidate.get('company', ''), derive_end_client(extract)):
        if name and (reason := await tools_module.company_blacklist_reason(name, context=blacklist_context)):
            return f'blacklisted company: {name} ({reason})'

    if extract['closed'] or 'no longer accepting applications' in full_text.lower():
        return 'posting closed (no longer accepting applications)'
    posted = parse_posting_date(extract['date_posted'] or candidate['date_posted'])
    if posted and (date.today() - posted).days > JOB_STALE_AGE_DAYS:
        return f'posting older than {JOB_STALE_AGE_DAYS} days ({posted.isoformat()})'
    sponsorship_locations = preferences.sponsorship_required_in()
    location = extract['location'].lower()
    needs_sponsorship = any(loc in location for loc in sponsorship_locations)
    if sponsorship_locations and (
        _requires_current_us_auth(full_text) or (needs_sponsorship and 'sponsor' not in full_text.lower())
    ):
        return 'job without explicit visa sponsorship'
    known_languages = preferences.languages()
    if known_languages:
        unsupported = [
            lang.strip() for lang in re.split(r'[,;/]', extract.get('language_requirement', ''))
            if lang.strip() and not any(known in lang.lower() for known in known_languages)
        ]
        if unsupported:
            return f'requires unsupported language: {", ".join(unsupported)}'
    degree = derive_education_requirement(extract)
    if degree and degree in preferences.rejected_degrees():
        return f'requires advanced degree: {degree}'
    return None


def hybrid_location_is_acceptable(location: str) -> bool:
    """True if a hybrid/on-site role in this location is one the user would actually take."""
    haystack = (location or '').lower()
    return any(token in haystack for token in preferences.hybrid_acceptable_locations())


def apply_rating_caps(extract: dict, rating: int) -> tuple[int, str]:
    """Deterministic post-rating ceiling. Returns (rating, reason) — reason is '' if uncapped.

    A cap is not a rejection: the job is still saved and still appears in the audit log, it just
    never crosses the >=4 notification threshold. This backstops the evaluator prompt, which has
    demonstrably rated a hybrid role in an unacceptable location 4/5 while naming the hybrid
    location as a drawback in its own reasoning.
    """
    workplace_type = derive_workplace_type(extract)
    cap = preferences.hybrid_rating_cap()
    if workplace_type in {'hybrid', 'onsite'} and not hybrid_location_is_acceptable(extract.get('location', '')):
        if rating > cap:
            location = extract.get('location') or 'unspecified location'
            return cap, f'{workplace_type} in {location} (not an acceptable hybrid location)'
    return rating, ''


_CONTRACT_RE = re.compile(
    r'\bcontract(?:or)?\b|\bfreelance\b|\bday\s*rate\b|\b\d+\s*-\s*\d+\s*months?\b|\bfixed[\s-]term\b',
    re.IGNORECASE,
)


def build_deterministic_warnings(candidate: dict, extract: dict) -> list[str]:
    """Warnings derived in code from the extract, independent of what the rater chose to report.

    The rater cannot be trusted to surface these on its own — it saw 'Hybrid - 2-3 days onsite'
    for the DevologyX posting and still rated it 4/5 without warning.
    """
    warnings: list[str] = []

    workplace_type = derive_workplace_type(extract)
    if workplace_type in {'hybrid', 'onsite'}:
        label = 'Hybrid' if workplace_type == 'hybrid' else 'On-site'
        location = extract.get('location') or 'location not stated'
        warning = f'{label} — {location}'
        if not hybrid_location_is_acceptable(extract.get('location', '')):
            warning += ' (not an acceptable hybrid location)'
        warnings.append(warning)

    if extract.get('relocation'):
        warnings.append(f"Relocation required: {extract['relocation']}")

    if _CONTRACT_RE.search(f"{extract.get('title', '')} {extract.get('description', '')[:3000]}"):
        warnings.append('Contract role — full-time preferred')

    if not extract.get('salary'):
        warnings.append('No salary listed')

    # Who is actually hiring changes how you apply, and the rater reports it only by luck: on the
    # 2026-08-11 run it flagged CyberCoders and Jobgether but not Hire Feed or Genius Innovation
    # Lab — 2 of the 4 agency postings that triggered a notification said nothing. Same reasoning
    # as the hybrid warning: a structural fact cannot be left to the rater's discretion.
    if derive_agency_posting(extract):
        end_client = derive_end_client(extract)
        warnings.append(
            f'Posted by a recruiting agency — hiring company: {end_client}' if end_client
            else 'Posted by a recruiting agency — actual hiring company not named'
        )

    return warnings


def merge_warnings(deterministic: list[str], llm_warnings: list[str]) -> list[str]:
    """Deterministic warnings first, then the rater's, dropping case-insensitive duplicates."""
    merged: list[str] = []
    seen: set[str] = set()
    for warning in [*deterministic, *llm_warnings]:
        text = str(warning).strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            merged.append(text)
    return merged


def _bullet_block(heading: str, items: list[str]) -> str:
    """A heading plus '• ' bullets, or '' when there is nothing to show."""
    if not items:
        return ''
    bullets = '\n'.join(f'• {item}' for item in items)
    return f'{heading}\n{bullets}'


def format_job_notification(
    candidate: dict, extract: dict, rating: int, pros: list[str], warnings: list[str]
) -> str:
    """The Telegram job-match message. Plain text — _send_pipeline_notification sends no
    parse_mode, so this uses bullets and emoji rather than Markdown."""
    header = f"⭐ {rating}/5 — {candidate['company']} — {candidate['title']}"
    location = extract.get('location', '').strip()
    workplace_type = derive_workplace_type(extract)
    if location and workplace_type and workplace_type not in location.lower():
        location = f'{location} — {workplace_type}'
    sections = [f'{header}\n📍 {location}' if location else header]
    for block in (_bullet_block('✅ Good:', pros), _bullet_block('⚠️ Warnings:', warnings)):
        if block:
            sections.append(block)
    sections.append(candidate['url'])
    return '\n\n'.join(sections)


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
            cost_delta = accumulate_stage_stats(stage_stats, msg)
            print_result_stats(msg, cost_delta)
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
    candidate: dict, rating: int, summary: str, content: str, notify: bool = True,
    extract: dict | None = None, pros: list[str] | None = None, warnings: list[str] | None = None,
) -> None:
    await tools_module.do_save_job_posting(
        company=candidate['company'], description=summary, rating=rating,
        content=content, job_id=candidate['job_id'],
    )
    if notify and rating >= 4:
        await _send_pipeline_notification(
            format_job_notification(candidate, extract or {}, rating, pros or [], warnings or [])
        )


def _hard_rule_category(reason: str) -> str:
    """Bucket a hard-rule reason string for the funnel summary."""
    if 'blacklisted' in reason:
        return 'hard_ruled_blacklisted'
    if 'closed' in reason:
        return 'hard_ruled_closed'
    if 'older than' in reason:
        return 'hard_ruled_stale'
    if 'sponsorship' in reason:
        return 'hard_ruled_us_auth'
    if 'language' in reason:
        return 'hard_ruled_language'
    if 'degree' in reason:
        return 'hard_ruled_education'
    return 'hard_ruled_other'


async def evaluate_all_candidates(
    candidates: list[dict], playwright_mcp: dict, evaluator_prompt: str,
    profile_block: str, stage_stats: dict, funnel: dict | None = None, audit: bool = False,
) -> None:
    """Stage 2: per candidate — Haiku extract, deterministic hard rules, local triage,
    then one configurable-model rating call. All sharing one browser.

    When ``audit`` is set, gate-killed candidates (hard-ruled / triaged-out) are STILL
    sent through the strong rater so we can detect false negatives (gate dropped it but
    the strong model rates it >=3). Normal saving/notification behaviour is unchanged.
    ``funnel`` (if provided) accumulates per-stage drop counts for the run summary.
    """
    if funnel is None:
        funnel = {}

    def bump(key: str) -> None:
        funnel[key] = funnel.get(key, 0) + 1

    async def audit_gate(candidate: dict, extract_text: str, gate_label: str, gate_detail: str) -> None:
        """In audit mode, re-rate a gate-killed job with the strong rater and flag false negatives."""
        if not audit:
            return
        try:
            result = await rate_job(evaluator_prompt, extract_text, stage_stats['rating'])
            strong = result['rating']
            verdict = 'FALSE NEGATIVE' if strong >= 3 else 'confirmed drop'
            if strong >= 3:
                bump('audit_false_negatives')
            logger.info(
                f"AUDIT [{gate_label}] {candidate['company']} — {candidate['title']}: "
                f"gate dropped ({gate_detail}) but strong rater says {strong}/5 → {verdict} — {result['reasoning']}"
            )
        except Exception as ex:
            logger.warning(f"AUDIT rating failed for {candidate['company']} — {candidate['title']}: {ex}")

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
                bump('extract_failed')
                tools_module.record_job_outcome(
                    candidate['site'], candidate['job_id'], 'extract_failed', summary='page extraction failed'
                )
                logger.warning(f"Extract failed (both paths): {candidate['company']} — {candidate['title']}")
                continue
            bump('extract_ok')
            extract_text = format_extract_text(candidate, extract)
            logger.info(
                f"Extract signal: {candidate['company']} — {candidate['title']}: "
                f"date_posted={extract.get('date_posted')!r} location={extract.get('location')!r} "
                f"closed={extract.get('closed')} language_requirement={extract.get('language_requirement')!r} "
                f"relocation={extract.get('relocation')!r} "
                f"education_requirement={extract.get('education_requirement')!r} "
                f"is_agency={derive_agency_posting(extract)} end_client={extract.get('end_client')!r}"
            )

            # Already-applied, checked against the END CLIENT only.
            #
            # Stage 1b's check ran against the poster's name, which for a recruiter is the agency —
            # so a recruiter reposting a role at a company the user already applied to directly
            # sails through. The end client is only knowable after the extract, hence here.
            #
            # Deliberately NOT extended to the agency's own name: applying once through a staffing
            # firm must never suppress every other company it posts for. That is the whole reason
            # load_applied_jobs() keeps agency names out of _applied_companies
            # (tools_generic.py:486-491), and re-adding it here would undo it. Agency status also
            # never changes the rating and never rejects on its own.
            end_client = derive_end_client(extract)
            if end_client:
                if matched_pdf := await tools_module.company_matches_applied(end_client):
                    bump('end_client_already_applied')
                    logger.info(
                        f"Already applied via end client: {candidate['company']} — "
                        f"{candidate['title']}: hiring company {end_client!r} matches {matched_pdf}"
                    )
                    await _save_and_notify(
                        candidate, rating=1,
                        summary=f"already applied to end client {end_client}",
                        content=(
                            f"# Already applied — posted by {candidate['company']} on behalf of "
                            f"{end_client}\n\nMatched applied-job record: {matched_pdf}\n\n{extract_text}"
                        ),
                        notify=False,
                    )
                    tools_module.record_job_outcome(
                        candidate['site'], candidate['job_id'], 'already_applied', rating=1,
                        summary=f'end client {end_client} already applied to',
                    )
                    continue

            hard_rule_reason = await apply_hard_rules(candidate, extract)
            if hard_rule_reason:
                bump(_hard_rule_category(hard_rule_reason))
                logger.info(f"Hard rule: {candidate['company']} — {candidate['title']}: rated 1 ({hard_rule_reason})")
                await _save_and_notify(
                    candidate, rating=1, summary=f"auto rejected {hard_rule_reason}",
                    content=f"# Auto-rated 1 — {hard_rule_reason}\n\n{extract_text}", notify=False,
                )
                tools_module.record_job_outcome(
                    candidate['site'], candidate['job_id'], 'hard_ruled', rating=1, summary=hard_rule_reason
                )
                await audit_gate(candidate, extract_text, 'hard_rule', hard_rule_reason)
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
                    bump('triaged_out')
                    await _save_and_notify(
                        candidate, rating=triage_result['score'],
                        summary=f"triaged out {triage_result['reason']}",
                        content=(
                            f"# Triaged out (local score {triage_result['score']}/5)\n\n"
                            f"Reason: {triage_result['reason']}\n\n{extract_text}"
                        ),
                        notify=False,
                    )
                    tools_module.record_job_outcome(
                        candidate['site'], candidate['job_id'], 'triaged_out',
                        rating=triage_result['score'], summary=triage_result['reason'],
                    )
                    await audit_gate(
                        candidate, extract_text, 'triage', f"local score {triage_result['score']}"
                    )
                    continue

            result = await rate_job(evaluator_prompt, extract_text, stage_stats['rating'])
            rating, cap_reason = apply_rating_caps(extract, result['rating'])
            if cap_reason:
                bump('rating_capped')
                logger.info(
                    f"Rating capped: {candidate['company']} — {candidate['title']}: "
                    f"{result['rating']} → {rating} ({cap_reason})"
                )
            bump(f'rated_{rating}')
            triage_note = f" (triage said {triage_result['score']})" if triage_result else ''
            logger.info(
                f"Rating: {candidate['company']} — {candidate['title']}: "
                f"{rating}/5 via {RATING_PROVIDER}{triage_note} — {result['reasoning']}"
            )
            pros = [str(p) for p in (result.get('pros') or [])]
            warnings = merge_warnings(
                build_deterministic_warnings(candidate, extract),
                [str(w) for w in (result.get('warnings') or [])],
            )
            content = (
                f"# {result['title']} at {result['company']} — rating {rating}/5\n\n"
                + (f"⚠ rating capped from {result['rating']}: {cap_reason}\n\n" if cap_reason else '')
                + (_bullet_block('## Good', pros) + '\n\n' if pros else '')
                + (_bullet_block('## Warnings', warnings) + '\n\n' if warnings else '')
                + f"Reasoning: {result['reasoning']}\n"
                + (f"Triage score (local): {triage_result['score']}\n" if triage_result else '')
                + f"\n{extract_text}"
            )
            tools_module.record_job_outcome(
                candidate['site'], candidate['job_id'], 'rated',
                rating=rating, summary=result['reasoning'],
            )
            await _save_and_notify(
                candidate, rating=rating, summary=result['summary'], content=content,
                extract=extract, pros=pros, warnings=warnings,
            )
        except Exception as ex:
            bump('eval_error')
            tools_module.record_job_outcome(
                candidate['site'], candidate['job_id'], 'eval_error', summary=str(ex)[:200]
            )
            console.print(f"[red]Stage 2 error evaluating {candidate['company']} — {candidate['title']}: {ex}[/red]")
            logger.warning(f"Stage 2 error evaluating {candidate['company']} — {candidate['title']}: {ex}")


async def _rate_with_opus(evaluator_prompt: str, extract_text: str, stage_stats: dict) -> dict:
    """One non-agentic Opus rating call — the reference standard for audits only."""
    options = ClaudeAgentOptions(
        tools=[],
        system_prompt=evaluator_prompt,
        permission_mode='bypassPermissions',
        model=MODEL_NAME_HIGH,
        output_format={
            'type': 'json_schema',
            'schema': {
                'type': 'object',
                'properties': {
                    'rating': {'type': 'integer', 'description': 'Fit rating 1-5'},
                    'company': {'type': 'string'},
                    'title': {'type': 'string'},
                    'reasoning': {'type': 'string', 'description': '2-3 sentences on the fit'},
                    'summary': {'type': 'string'},
                    # Optional here: the audit path reads only rating/reasoning, but the shared
                    # evaluator prompt asks for these, so the schema must allow them.
                    'pros': {'type': 'array', 'items': {'type': 'string'}},
                    'warnings': {'type': 'array', 'items': {'type': 'string'}},
                },
                'required': ['rating', 'company', 'title', 'reasoning', 'summary'],
            },
        },
        cwd=str(PROJECT_DIR),
    )
    async for msg in sdk_query(prompt=f'Evaluate this job posting:\n\n{extract_text}', options=options):
        if isinstance(msg, ResultMessage):
            accumulate_stage_stats(stage_stats, msg)
            if msg.structured_output:
                return msg.structured_output
    raise RuntimeError('Opus audit rating returned no structured output')


async def audit_unsurfaced_with_opus(
    playwright_mcp: dict, evaluator_prompt: str, stage_stats: dict, sample_size: int,
) -> list[dict]:
    """Sample jobs the pipeline never surfaced and re-rate them with Opus.

    Three pools (filtered at Stage 1, seen but never queued, mid-rated 2-3). A high Opus
    rating on any of them is a FALSE NEGATIVE: a job the funnel should have delivered.
    Diagnostic only — nothing is saved or notified.
    """
    pools = tools_module.unsurfaced_pools()
    findings: list[dict] = []

    for pool_name, records in pools.items():
        sample = records[:sample_size]
        if not sample:
            logger.info(f'Opus audit: pool {pool_name!r} is empty, nothing to sample')
            continue
        logger.info(f'Opus audit: sampling {len(sample)} of {len(records)} job(s) from pool {pool_name!r}')
        for record in sample:
            candidate = {
                'site': record['site'], 'job_id': record['job_id'], 'url': record['url'],
                'title': record['title'], 'company': record['company'],
                'date_posted': record.get('date_posted') or '', 'snippet': record.get('snippet', ''),
            }
            try:
                extract = await extract_job_page_direct(candidate, playwright_mcp['url'], stage_stats['extraction'])
                if extract is None:
                    logger.warning(f'Opus audit: extract failed for {record["company"]} — {record["title"]}')
                    continue
                result = await _rate_with_opus(evaluator_prompt, format_extract_text(candidate, extract), stage_stats['rating'])
            except Exception as ex:
                logger.warning(f'Opus audit failed for {record["company"]} — {record["title"]}: {ex}')
                continue

            verdict = 'FALSE NEGATIVE' if result['rating'] >= 4 else 'confirmed drop'
            findings.append({
                'pool': pool_name, 'company': record['company'], 'title': record['title'],
                'url': record['url'], 'opus_rating': result['rating'],
                'verdict': verdict, 'reasoning': result['reasoning'],
            })
            logger.info(
                f'Opus audit [{pool_name}] {record["company"]} — {record["title"]}: '
                f'Opus rates {result["rating"]}/5 → {verdict} — {result["reasoning"]}'
            )

    false_negatives = sum(1 for f in findings if f['verdict'] == 'FALSE NEGATIVE')
    logger.info(f'Opus audit summary: {len(findings)} job(s) re-rated, {false_negatives} false negative(s)')
    return findings


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
        '--output-dir', str(PLAYWRIGHT_OUTPUT_DIR),
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


async def run_non_interactive(browser_mode: str = 'headless', audit: bool = False, audit_opus: int = 0) -> None:
    tools_module._candidates = []
    tools_module._candidates_per_query = {}
    tools_module._job_extracts = []
    tools_module._check_status_counts = {}
    tools_module._listing_records = {}
    tools_module._queries_searched = {}
    tools_module._check_status_per_query = {}
    tools_module._queue_skipped_counts = {}
    tools_module._current_query = None
    funnel: dict[str, int] = {}
    audit_findings: list[dict] = []
    if audit:
        console.print('[bold magenta]AUDIT mode: gate-killed jobs will still be re-rated to detect false negatives.[/bold magenta]')

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
                "stage_stats": public_stage_stats(stage_stats),
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
            # Removes these from the model's context entirely. allowed_tools would NOT — it only
            # auto-grants permission, which bypassPermissions already does. See the constant.
            disallowed_tools=SCRAPER_DISALLOWED_BROWSER_TOOLS,
            cwd=str(PROJECT_DIR),
            model=MODEL_NAME_LOW,
            max_turns=SCRAPER_MAX_TURNS_PER_QUERY,
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
            # check_status distinguishes dedup-saturation (all already_processed) from an
            # authwall/empty-page (nothing seen at all) — the two zero-candidate root causes.
            check_status_counts = dict(tools_module._check_status_counts)
            logger.warning(
                f"Run funnel (0 candidates): listings_seen={sum(check_status_counts.values())} "
                f"check_status={json.dumps(check_status_counts)}"
            )
            query_lines = "\n".join(f'  - "{q}": {candidates_per_query.get(q, 0)} jobs' for q in queries)
            await _send_pipeline_notification(
                f"Job search run FAILED\n• Error: 0 candidates found\n• Queries ({len(queries)}):\n{query_lines}\n• Listings seen: {sum(check_status_counts.values())} {check_status_counts}\n• Elapsed: {elapsed_mins:.1f} min\n• Total cost: ${total_cost:.4f}"
            )
            log_run_cost({
                "timestamp": datetime.now().isoformat(),
                "mode": "non-interactive",
                "status": "failed_no_candidates",
                "stage_stats": public_stage_stats(stage_stats),
                "total_cost": total_cost,
                "check_status": check_status_counts,
                "elapsed_minutes": elapsed_mins,
            })
            # A zero-candidate run is exactly when the trace matters most — it distinguishes
            # "no query was ever searched" from "searched, everything deduped".
            audit_log_path = tools_module.write_run_audit_log(
                queries=queries,
                applied_jobs_in_horizon=len(tools_module._applied_jobs),
                applied_jobs_total=len(list(tools_module.APPLIED_JOBS_DIR.glob('*.pdf'))),
                funnel={
                    "queries_generated": len(queries),
                    "listings_seen": sum(check_status_counts.values()),
                    "check_status": check_status_counts,
                    "check_status_per_query": dict(tools_module._check_status_per_query),
                    "queue_skipped": dict(tools_module._queue_skipped_counts),
                    "candidates_queued": 0,
                },
            )
            console.print(f"[dim]Run audit log: {audit_log_path}[/dim]")
            return

        # Stage 2: all evaluations share the same browser via the SSE server
        console.print("[yellow]Stage 2: Evaluating candidates ...[/yellow]\n")
        reference_block = await build_reference_summary(stage_stats["reference_summary"])
        evaluator_prompt = build_evaluator_prompt(reference_block)
        profile_block = build_profile_block(reference_block)
        await evaluate_all_candidates(
            candidates, playwright_mcp, evaluator_prompt, profile_block, stage_stats,
            funnel=funnel, audit=audit,
        )
        if audit_opus:
            console.print(f"[bold magenta]Opus audit: sampling up to {audit_opus} un-surfaced job(s) per pool ...[/bold magenta]")
            audit_findings = await audit_unsurfaced_with_opus(
                playwright_mcp, evaluator_prompt, stage_stats, audit_opus
            )
            funnel['audit_opus_false_negatives'] = sum(
                1 for f in audit_findings if f['verdict'] == 'FALSE NEGATIVE'
            )
    finally:
        playwright_proc.terminate()
        await playwright_proc.wait()

    elapsed_mins = (time.time() - start_time) / 60
    num_evaluated, num_high_rated = count_new_jobs(jobs_before)
    total_cost = sum(s["cost"] for s in stage_stats.values())

    check_status_counts = dict(tools_module._check_status_counts)
    funnel_summary = {
        "queries_generated": len(queries),
        "listings_seen": sum(check_status_counts.values()),
        "check_status": check_status_counts,
        "check_status_per_query": dict(tools_module._check_status_per_query),
        "queue_skipped": dict(tools_module._queue_skipped_counts),
        "candidates_queued": len(candidates),
        **funnel,
    }
    logger.info(f"Run funnel: {json.dumps(funnel_summary)}")
    audit_log_path = tools_module.write_run_audit_log(
        queries=queries,
        applied_jobs_in_horizon=len(tools_module._applied_jobs),
        applied_jobs_total=len(list(tools_module.APPLIED_JOBS_DIR.glob('*.pdf'))),
        funnel=funnel_summary,
        audit_findings=audit_findings,
    )
    console.print(f"[dim]Run audit log: {audit_log_path}[/dim]")
    if audit:
        logger.info(f"AUDIT summary: {funnel.get('audit_false_negatives', 0)} false negative(s) "
                    "(gate-dropped but strong rater >=3)")

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
        "stage_stats": public_stage_stats(stage_stats),
        "total_cost": total_cost,
        "candidates_found": len(candidates),
        "jobs_saved": num_evaluated,
        "jobs_rated_high": num_high_rated,
        "funnel": funnel_summary,
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
    # One 'HTTP Request: ...' line per MCP tool call drowns out our own logging; set
    # HTTP_LOG_LEVEL=INFO to get the transport chatter back when debugging a tool server.
    http_log_level = os.environ.get('HTTP_LOG_LEVEL', 'WARNING')
    for noisy_logger_name in ('httpx', 'httpcore', 'mcp.client.streamable_http'):
        logging.getLogger(noisy_logger_name).setLevel(http_log_level)
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
    parser.add_argument(
        "--status",
        action="store_true",
        help="Report whether a job search run is currently active, then exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Start even if another run holds the run lock (concurrent runs steal each other's jobs)",
    )
    parser.add_argument(
        "--audit-opus",
        nargs="?",
        type=int,
        const=AUDIT_OPUS_SAMPLE_SIZE,
        default=0,
        metavar="N",
        help=f"Diagnostic: sample N jobs (default {AUDIT_OPUS_SAMPLE_SIZE}) from each un-surfaced pool "
             "(filtered at Stage 1 / seen but never queued / rated 2-3) and re-rate them with Opus "
             "to find jobs the funnel should have delivered",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Diagnostic: re-rate gate-killed jobs with the strong rater to detect false negatives "
             "(non-interactive mode only; saving/notification behaviour unchanged)",
    )
    args = parser.parse_args()
    interactive = not args.non_interactive

    if args.status:
        console.print(tools_module.describe_run_status())
        return

    # Concurrent runs share processed_jobs/ and one browser profile: the first run to see a
    # listing marks it processed, so the second dedups it away and neither evaluates it.
    conflict = tools_module.acquire_run_lock(
        mode='interactive' if interactive else 'non-interactive', force=args.force,
    )
    if conflict:
        console.print(
            f"[red]A job search run is already active (pid {conflict['pid']}, started "
            f"{conflict['started_at']}, {conflict['mode']}).[/red]\n"
            f"[red]Command: {conflict['argv']}[/red]\n"
            "[yellow]Concurrent runs steal each other's jobs via processed_jobs/ and fight over "
            "the browser. Wait for it to finish, or re-run with --force to override.[/yellow]"
        )
        sys.exit(1)

    if not interactive and not JOB_REQUIREMENTS_PATH.exists():
        console.print("[red]Error: JOB_REQUIREMENTS.md not found. Non-interactive mode requires it.[/red]")
        sys.exit(1)

    # Release on every exit path — a lock left behind by a crash would block the next run
    # until someone noticed the stale file (read_run_lock also treats a dead PID as stale).
    try:
        load_processed_jobs()
        await tools_module.categorize_downloads_pdfs()
        await tools_module.ingest_downloads_applied_pdfs()
        await tools_module.load_applied_jobs()

        if interactive:
            options = ClaudeAgentOptions(
                system_prompt=build_system_prompt(interactive=True),
                mcp_servers={
                    "playwright": {
                        "type": "stdio",
                        "command": "npx",
                        "args": [
                            "@playwright/mcp@latest",
                            "--user-data-dir", str(BROWSER_PROFILE_DIR),
                            "--output-dir", str(PLAYWRIGHT_OUTPUT_DIR),
                        ],
                    },
                    "job_search": make_job_search_server(interactive=True),
                },
                permission_mode="acceptEdits",
                cwd=str(PROJECT_DIR),
            )
            async with ClaudeSDKClient(options) as client:
                await run_interactive(client)
        else:
            await run_non_interactive(browser_mode=args.browser, audit=args.audit, audit_opus=args.audit_opus)
    finally:
        tools_module.release_run_lock()


def cli() -> None:
    asyncio.run(main())
