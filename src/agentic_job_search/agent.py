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
from collections.abc import Sequence
from typing import Any

from utils_tools_n_agents_common.logging_setup import setup_logging
from utils_tools_n_agents_common.models import (
    ANTHROPIC_MODEL_NAME_HIGH,
    ANTHROPIC_MODEL_NAME_LOW,
    ANTHROPIC_MODEL_NAME_MEDIUM,
    route_for,
)

from agentic_job_search.config import (
    APPLIED_JOBS_HORIZON_DAYS,
    AUDIT_OPUS_SAMPLE_SIZE,
    JOB_STALE_AGE_DAYS,
    MAX_REFERENCE_JOBS,
    MAX_SEARCH_QUERIES,
    MODEL_NAME_EXTRACTOR,
    MODEL_NAME_QUERY,
    MODEL_NAME_RATING,
    MODEL_NAME_SCRAPER,
    PLAYWRIGHT_MCP_PACKAGE,
    PLAYWRIGHT_MCP_REGISTRY_URL,
    PLAYWRIGHT_MCP_VERSION,
    PLAYWRIGHT_MCP_VERSION_CHECK_TIMEOUT_SECONDS,
    SCRAPER_DISALLOWED_BROWSER_TOOLS,
    SCRAPER_INTER_QUERY_DELAY_SECONDS,
    SCRAPER_DATE_POSTED_LABEL,
    SCRAPER_EXPERIENCE_LABEL,
    SCRAPER_INTER_ACTION_DELAY_SECONDS,
    SCRAPER_INTER_SEARCH_DELAY_SECONDS,
    REGION_OVERLAP_ALERT_THRESHOLD,
    SATURATION_MIN_NEW_JOBS,
    SATURATION_MIN_NEW_RATIO,
    SCRAPER_MAX_LISTINGS_PER_SEARCH,
    SCRAPER_MAX_TURNS_PER_QUERY,
    SCRAPER_MIN_TURNS_PER_QUERY,
    SCRAPER_MIN_LISTINGS_PER_QUERY,
    REFERENCE_SUMMARY_MAX_CHARS,
    THINKING_MAX_CHARS,
    TRIAGE_ENABLED,
)
from agentic_job_search.extract_openrouter import extract_job_page_openrouter
from agentic_job_search import location
from agentic_job_search import location_review
from agentic_job_search.location import classify_location, is_eu_member, location_token_matches
from agentic_job_search import scrape_openrouter
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
    ProviderUnavailableError,
    call_mcp_tool,
    chat_openrouter,
    extract_json_object,
    generate_local,
    mcp_session,
    preflight_local_model,
    rate_with_ollama,
    rate_with_openrouter,
    triage_job_fit,
    triage_rejects,
    unwrap_exception,
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

# Structural report on the search page, run once per search and handed to `report_search`, which
# does the JUDGING in code. The model is not asked whether the page "looks right": that is exactly
# the discretion that let a dead EU region and a silently-restructured results list run for days.
#
# Read-only, like SCRAPER_HARVEST_JS. It reports what exists, never acts on it.
SCRAPER_UI_CONTRACT_JS = r"""() => {
  const box = document.querySelector('div[componentkey="SearchResultsMainContent"]');
  const txt = el => ((el && el.textContent) || '').replace(/\s+/g, ' ').trim();

  // The search box is a plain <input> carrying a PLACEHOLDER, not an aria-label — verified
  // against the live page. Matching on aria-label alone reported it missing on every search.
  const searchBox = [...document.querySelectorAll('input[type="text"], input:not([type]), textarea')]
    .some(el => /describe the job|search by title|job title|keyword/i.test(
      (el.getAttribute('placeholder') || '') + ' ' + (el.getAttribute('aria-label') || '')));

  // Cards and chips are each rendered TWICE in the DOM (the harvest JS dedupes cards for the
  // same reason), so a raw count reads 50 cards and 44 chips on a 25-job, 22-chip page.
  const cardIds = new Set();
  for (const card of document.querySelectorAll('div[componentkey^="job-card-component-ref-"]')) {
    const id = (card.getAttribute('componentkey') || '').replace('job-card-component-ref-', '');
    if (id) cardIds.add(id);
  }

  // Chips carry an accessible name "Filter by <label>"; once a value is chosen the label BECOMES
  // the value ("Past week", "Senior"), which is how a click is confirmed without ever crafting
  // the filter URL ourselves.
  const chips = [...new Set([...document.querySelectorAll('[aria-label^="Filter by "], [title^="Filter by "]')]
    .map(el => (el.getAttribute('aria-label') || el.getAttribute('title') || '')
      .replace(/^Filter by\s*/, '').trim())
    .filter(Boolean))];

  // The location pin sits beside the result count, labelled "Location <region>".
  const locEl = document.querySelector('[aria-label^="Location "], [title^="Location "]');
  const locationPin = locEl
    ? (locEl.getAttribute('aria-label') || locEl.getAttribute('title') || '')
        .replace(/^Location\s*/, '').trim()
    : '';

  const resultCountEl = [...document.querySelectorAll('p, span, h2')]
    .find(el => /^\d[\d,+]*\s+results?$/i.test(txt(el)));

  // Pagination footer: a "Next" control alongside numbered pages.
  const pagination = [...document.querySelectorAll('a, button')]
    .some(el => /^next\b/i.test(txt(el)));

  return {
    url: location.href,
    search_box: searchBox,
    results_container: !!box,
    job_cards: cardIds.size,
    chip_row: chips.length,
    chips: chips,
    location_pin: locationPin,
    pagination: pagination,
    result_count_text: resultCountEl ? txt(resultCountEl) : '',
    body_sample: txt(document.body).slice(0, 1500).toLowerCase(),
  };
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

**1. Filters live in the chip row, not in the URL you type.** The search box is a
natural-language field labelled "Describe the job you want", and LinkedIn rewrites whatever you put
in it. Above the results is a row of filter chips — Date posted, Experience level, Employment type,
Company — plus a location chip beside the result count. **That chip row is how filters get set.**

Typing the region into the query text does NOT work: it is silently ignored, and for several days
every "European Union" search quietly returned the account's home metro instead. Crafting
`geoId=`/`f_TPR=` URLs *does* work, and is forbidden anyway — see Step 1.

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

Put ONLY the query text in `keywords` — the job title plus the word `remote`. A bare keyword URL
like this is what a bookmarked or shared LinkedIn search looks like, so arriving this way is
ordinary. The empty `f_SAL=` clears a leftover salary filter that persists between sessions and
otherwise silently narrows every search.

**Never put `geoId`, `f_TPR`, `f_E`, `f_WT` or any other filter parameter in a URL you navigate
to.** They do work — that is not the point. No human assembles filter parameters by hand, and
doing so is a cheap signal that this is not a person. Filters get clicked, in Step 2.

### Step 2 — set the filters by clicking the chips, like a person

Above the results is a row of filter chips. Take a snapshot, then work through the filters listed
for this search under "Search coverage" below, **pausing {action_delay_min}–{action_delay_max}
seconds between each interaction** (use browser_wait_for, and vary it — never the same gap twice).

For the **location**: click the location chip (it shows the current region next to the result
count), clear it, type the region name, then pick the matching suggestion from the autocomplete.
For **Date posted** and **Experience level**: click the chip, click the option, then click
"Show results".

`target` takes the **bare ref exactly as the snapshot shows it**, e.g. `target: "e1202"` — not
`[ref=e1202]`, not the snapshot's display text like `button "Location"`, and there is no `ref`
parameter. A unique CSS selector also works. If a click fails, re-read the snapshot and use the
bare ref rather than reformatting the same guess.

Two things to keep straight:

- These chips are ABOVE the results list. Clicking them is fine and expected. The **results list
  itself is still never clicked** — see the Dismiss warning above. If you cannot find a chip,
  say so and stop; do not go hunting through the result cards for it.
- After choosing a value, a chip's label changes from "Date posted" to "Past week", and from
  "Experience level" to "Senior". That relabelling is how you know the click landed.

Two things that look wrong and are not:

- After "Show results", LinkedIn puts a salary value back into the URL
  (`f_SAL=f_SA_id_…`). That is LinkedIn's own account setting re-attaching itself. It is
  expected. **Do not navigate again to remove it.**
- **Once you have clicked any filter, do not navigate again during this search.** Navigating
  throws away every filter you clicked, but the location chip keeps *showing* the old region, so
  the page looks filtered when it is not. If you truly must start the search over, click every
  chip again — the location chip included, even when it already shows the right region.

### Step 3 — confirm what you are actually looking at

Call **run_ui_contract** with the region name for this search.

It runs the exact contract JavaScript for you, reads the result, and judges it in code — that
judgement is deliberately not yours to make. You do not write the JavaScript and you never handle
the report. Do what its answer tells you.

Also state in your reply the result count and the location the page is showing. If it says the
search is unusable, follow what it says and stop.

If you see a CAPTCHA, "unusual activity", a verification challenge, or a forced re-login: call
**report_problem** and stop the whole query. Do not retry it and do not try to work around it.

### Step 4 — harvest every listing

Call **harvest_listings** — it takes no arguments.

The system runs the exact harvest JavaScript against the results container and keeps the listings
itself; you get back a count. Job ids come from each card's `componentkey` attribute, so nothing in
the results list is ever clicked.

### Step 5 — record what was harvested

Call **record_listings** — it takes no arguments.

The system records from its own copy of the harvest. Deduplication, the age check, the
already-applied blocklist, title screening and queueing all happen in code, and it returns the
totals. Do not loop over listings, and do not describe, list, or retype job data.

If a harvest reports an error or zero jobs while the page visibly shows results, LinkedIn has
changed the markup: **call report_problem and stop**. Do not fall back to clicking the list. Do not
navigate to individual job pages — a separate evaluation agent visits them later.

### If you do not have real data, say so — that is a success, not a failure

You are never required to produce data you did not actually see. If a call fails, a result is
missing or unreadable, or the page is not what you expected, call **report_problem** and stop.
Reconstructing or filling in a plausible-looking result is the worst thing you can do here, because
downstream it is indistinguishable from a real one.

## Search coverage

You are given ONE search query per session. Run it as {search_count} searches — one per target
region below — then stop. Do not invent additional queries. For each search, set these filters by
clicking, in this order:

{search_list}

Between searches, wait {search_delay_min}–{search_delay_max} seconds.

Expect many `already_processed` results, especially on later searches. **That is expected and
correct — it is not a failure, and not a reason to skip the rest of a search.** The few new jobs
that come back are usually the best-matching ones you will find.

**But if two searches return the IDENTICAL list of jobs, the location filter did not take effect.**
That is a failure, not a sign the query is exhausted. Say so explicitly rather than stopping early.
Code checks this too, so do not be tempted to smooth it over.

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
- posting_language: the language the POSTING PAGE ITSELF IS WRITTEN IN, lowercase English name, e.g. "english", "french", "german". Judge the SOURCE page you read, NOT the condensed English text you are about to write — you translate as you condense, so your own output says nothing about the original. The original job title is usually the clearest tell (e.g. a title like "Scientifique principal des données en IA" means "french"). Leave empty only if genuinely undeterminable.
- residency_scope: "country_only" if the posting requires LIVING IN the country it is advertised in (e.g. "Remote within country", "must be based in Germany", "open only to candidates residing in Poland"), or "area_wide" if it offers a whole multi-country area (e.g. "remote anywhere in the EU", "Work from Anywhere", "any EMEA country"). Leave empty when the posting does not say. This is about where the HOLDER MUST LIVE, which is not the same as where the job is advertised: "Romania (Remote)" on its own says nothing here.
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

## Language
Check the `Posting written in:` and `Implied local language:` lines. A posting written in another language, and a workplace whose implied local working language is not English, are both real frictions — factor them into the rating even when the extract you are reading has been translated into English. Do not write a warning bullet for either: both are detected deterministically and added for you.

Do NOT write a warning about the poster being a recruiting agency or the hiring company being undisclosed — that is detected deterministically and added for you, and repeating it just duplicates the bullet in different words. Being posted by an agency is **not** a reason to lower the rating; judge the role itself.
"""


def build_scraper_instructions() -> str:
    """Scraper prompt with the configured search regions injected as CHIP sequences.

    Regions are a personal preference (see preferences.py), so the search list is generated
    rather than hardcoded. With no regions configured, the query runs once with no location
    filter (whatever the account defaults to).

    Filters are applied by CLICKING the chip row, never by putting `geoId`/`f_TPR` in a URL.
    Both work; only one is something a person does. Hand-assembled filter parameters are a cheap
    fingerprint for anti-automation, and this drives a real logged-in account (see the Account
    safety requirement in docs/requirements.md).

    The region must NOT go into the query text. That was the pre-2026-08-18 design and LinkedIn
    silently ignored it: every "European Union" search returned the account's home metro for days,
    which is why `geo_id` now exists purely to verify that the click landed.

    Sort order is not a coverage axis — the UI exposes no sort control. Region is.
    """
    regions = preferences.search_regions()
    date_label = SCRAPER_DATE_POSTED_LABEL
    exp_label = SCRAPER_EXPERIENCE_LABEL

    def filters_for(location: str) -> str:
        steps = []
        if location:
            steps.append(f'set **Location** to `{location}` (click the location chip, clear it, '
                         f'type it, pick the suggestion)')
        steps.append(f'set **Date posted** to `{date_label}`')
        steps.append(f'set **Experience level** to `{exp_label}`')
        return '; then '.join(steps)

    entries: list[str] = []
    if regions:
        for region in regions:
            name = region.get('name') or region.get('linkedin_location', '')
            location = str(region.get('linkedin_location', ''))
            entries.append(f'**{name}** — search `<QUERY>, remote`, then {filters_for(location)}.')
    else:
        entries.append(f'**Unfiltered** — search `<QUERY>, remote`, then {filters_for("")}.')

    search_list = '\n'.join(f'{i}. {entry}' for i, entry in enumerate(entries, start=1))
    search_min, search_max = SCRAPER_INTER_SEARCH_DELAY_SECONDS
    action_min, action_max = SCRAPER_INTER_ACTION_DELAY_SECONDS
    return SCRAPER_INSTRUCTIONS_TEMPLATE.format(
        search_count=len(entries),
        search_list=search_list,
        search_delay_min=search_min,
        search_delay_max=search_max,
        action_delay_min=action_min,
        action_delay_max=action_max,
        max_listings=SCRAPER_MAX_LISTINGS_PER_SEARCH,
        harvest_js=SCRAPER_HARVEST_JS,
        contract_js=SCRAPER_UI_CONTRACT_JS,
    )


def build_evaluator_instructions() -> str:
    """Evaluator prompt with the configured sponsorship, relocation, and hybrid rules injected."""
    sponsorship_section = _sponsorship_prompt_section()
    note = preferences.relocation_note()
    relocation_section = f'\n## Relocation (applies to REMOTE roles only)\n{note}\n' if note else ''

    locations = preferences.would_commute_here()
    commute_cost = (
        '- An office — hybrid, on-site, on location, in-person — means a daily commute whose '
        'travel time is NOT compensated, and it means permanently living close enough to that '
        'office. It constrains where the candidate can live AND costs them hours. Treat that as a '
        'real, substantial drawback in its own right, not a formatting detail.\n'
    )
    if locations:
        joined = ', '.join(locations)
        hybrid_locations_section = commute_cost + (
            f'- The only places worth commuting to are: {joined}.\n'
            '- Hybrid or on-site anywhere else is **not a fit** however well the role itself '
            'matches. Never rate those 4 or 5.\n'
        )
    else:
        hybrid_locations_section = commute_cost + (
            '- There is nowhere the candidate would commute to: rate every hybrid or on-site role '
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
        'posting_language': {'type': 'string', 'description': "Language the SOURCE page is written in, lowercase e.g. 'english', 'french' — judge the original page, not your condensed English output; the original title is the clearest tell"},
        'residency_scope': {'type': 'string', 'enum': ['country_only', 'area_wide', ''],
                            'description': "Whether the posting pins residence to the country it is anchored in ('country_only') or offers a whole multi-country area ('area_wide'); empty when the posting does not say"},
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
    """Newest `*-resume-*.md` in run_dir, or None.

    Markdown only — a PDF resume is not read. The applied-job corpus is the dominant signal for
    query generation and the ideal-role profile, so converting the PDF has never been worth it.
    """
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
            console.print("[yellow]Warning: no resume file found matching *-resume-*.md[/yellow]")

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
    """Query generation via the OpenRouter MCP server. Raises on any failure."""
    content, cost_usd = await chat_openrouter(f'{prompt}\n\n{QUERY_JSON_INSTRUCTIONS}', model=MODEL_NAME_QUERY)
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
    if route_for(MODEL_NAME_QUERY) == 'openrouter':
        try:
            captured = await _generate_queries_openrouter(prompt, stage_stats)
            provider = 'openrouter'
        except Exception as ex:
            logger.warning(f'Stage 1a via OpenRouter failed, falling back to Anthropic: {ex}')

    if not captured:
        # SDK isolation. These three default to "load whatever the interactive CLI would": the
        # docstring is explicit that setting_sources=None loads ALL sources and that "project"
        # pulls in CLAUDE.md files, strict_mcp_config=False adds every user/global MCP server, and
        # skills=None is NOT "skills off". Measured 2026-08-21: that put this repo's 67KB CLAUDE.md
        # (~17K tokens) plus nine unrelated MCP servers into a Haiku scraping session, re-read on
        # every turn. Nothing here reads project settings at runtime.

        options = ClaudeAgentOptions(
            setting_sources=[],
            strict_mcp_config=True,
            skills=[],
            tools=[],
            model=ANTHROPIC_MODEL_NAME_MEDIUM,
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
        setting_sources=[],
        strict_mcp_config=True,
        skills=[],
        model=ANTHROPIC_MODEL_NAME_LOW,
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
        f'cost_charged={cost_delta if cost_delta is not None else msg.total_cost_usd} '
        # is_error with subtype="success" is an HTTP-level API failure whose status lives in
        # api_error_status; without these three fields an errored result is indistinguishable
        # from a good one in the log, which is how "error result: success" stayed undiagnosable.
        f'is_error={getattr(msg, "is_error", None)} subtype={getattr(msg, "subtype", None)!r} '
        f'api_error_status={getattr(msg, "api_error_status", None)}'
    )
    if getattr(msg, 'is_error', False):
        logger.error(
            f'API error on session {msg.session_id}: status='
            f'{getattr(msg, "api_error_status", None)} subtype={getattr(msg, "subtype", None)!r} '
            f'terminal_reason={getattr(msg, "terminal_reason", None)!r} '
            f'errors={getattr(msg, "errors", None)}'
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
    """Total job listings the scraper has inspected this run (any check_and_record_job outcome).

    This counts CALLS. When two regions return the same cards it double-counts, which is exactly
    what made a collapsed region axis look like healthy coverage — see _listings_distinct.
    """
    return sum(tools_module._check_status_counts.values())


def _listings_distinct() -> int:
    """Distinct job listings inspected this run, regardless of how many searches surfaced them."""
    return len(tools_module._distinct_listing_ids)


def _distinct_snapshot() -> set[str]:
    """Copy of the distinct-id set, for before/after per-query diffing."""
    return set(tools_module._distinct_listing_ids)


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


def _same_query(reported: str, base: str) -> bool:
    """True if a report_search record belongs to this base query.

    The model reports the full search text it typed ("Principal Data Scientist, remote"), not the
    base query, so an exact match silently never fires.
    """
    reported = (reported or '').strip().lower()
    base = (base or '').strip().lower()
    return bool(base) and (reported == base or reported.startswith(base))


def assess_run_health(funnel: dict) -> list[str]:
    """Return human-readable alerts about THIS run's discovery quality; empty means healthy.

    A saturated run and a healthy one looked identical to the user: silence. Aug 15-18 2026
    produced 4, 4, 2 and 1 new jobs from ~300 listings each, four days in a row with no
    notification and no signal that anything had changed. The causes are separable and each is
    actionable, so the alert names which one applies rather than just saying "quiet run".
    """
    alerts: list[str] = []

    # 1. A changed page or a block. Most decisive, so it comes first.
    for alert in funnel.get('ui_alerts') or []:
        kind = alert.get('kind')
        where = f"{alert.get('query', '?')} / {alert.get('region', '?')}"
        if kind == 'blocked':
            alerts.append(f'BLOCKED by LinkedIn on {where}: "{alert.get("detail")}" — run backed off, did not retry')
        elif kind == 'contract':
            alerts.append(f'UI CHANGED on {where}: {alert.get("detail")}')
        elif kind == 'filters':
            if alert.get('resolved'):
                # Re-clicked and confirmed by a later passing report before harvest; audit §1b
                # still lists both reports, so the fail -> ok trail is not lost.
                continue
            alerts.append(f'FILTERS DID NOT APPLY on {where}: {alert.get("detail")}')
        elif kind == 'drift':
            alerts.append(f'UI drift on {where}: {alert.get("detail")}')
        elif kind == 'low_listings':
            alerts.append(f'LOW LISTING COUNT on {alert.get("query", "?")}: {alert.get("detail")}')
        elif kind == 'unverified_search':
            alerts.append(
                f'SEARCH NOT VERIFIED on "{alert.get("query", "?")}" ({alert.get("region")}): '
                f'{alert.get("detail")}'
            )
        elif kind == 'stopped_early':
            alerts.append(f'STOPPED EARLY on {alert.get("query", "?")}: {alert.get("detail")}')
        elif kind == 'local_model_missing':
            alerts.append(f'LOCAL TRIAGE DISABLED: {alert.get("detail")}')
        elif kind == 'playwright_mcp_outdated':
            alerts.append(f'BROWSER TOOLCHAIN OUTDATED: {alert.get("detail")}')
        elif kind == 'location_recommendations':
            alerts.append(f'COUNTRY LISTS: {alert.get("detail")}')

    # 1b. Queries that failed outright. Deliberately ahead of saturation: saturation is measured
    #     against DISTINCT listings and is skipped entirely when there are none (`if distinct:`
    #     below), so the case where something is most obviously wrong -- every query erroring, zero
    #     listings -- produced no alert at all. On 2026-09-02 all six queries died on an OpenRouter
    #     402 and audit section 1b still read "No alerts: page structure sound, regions distinct,
    #     yield acceptable".
    failed = funnel.get('queries_failed') or {}
    if failed:
        total = funnel.get('queries_generated') or len(failed)
        first_reason = str(next(iter(failed.values()), ''))[:300]
        if len(failed) >= total:
            alerts.append(
                f'ALL {total} QUERIES FAILED — no search completed, so 0 listings is not "nothing '
                f'new today". First error: {first_reason}'
            )
        else:
            # A partial failure is not the 2026-09-02 shape. On 2026-09-10 one 502 cost one query
            # its second region while the other five ran normally (156 distinct listings), and the
            # all-failed wording above told the reader that no search had completed.
            names = list(failed)
            shown = ', '.join(names[:3]) + (f' +{len(names) - 3} more' if len(names) > 3 else '')
            alerts.append(
                f'{len(failed)} of {total} QUERIES FAILED ({shown}) — their unfinished searches '
                f'did not run; the other {total - len(failed)} completed. First error: {first_reason}'
            )

    # 2. A dead region axis: two regions returning the same jobs is not "the query is exhausted".
    for query, overlap in (funnel.get('region_overlap') or {}).items():
        if overlap >= REGION_OVERLAP_ALERT_THRESHOLD:
            alerts.append(
                f'REGION OVERLAP {overlap:.0%} on "{query}" — the location filter is not '
                'separating regions, so half the searches are duplicates'
            )

    # 3. Saturation. Measured against DISTINCT listings; the call count double-counts a job seen
    #    in two regions and would flatter the ratio.
    distinct = funnel.get('listings_distinct') or 0
    new_jobs = (funnel.get('check_status') or {}).get('new', 0)
    if distinct:
        ratio = new_jobs / distinct
        if new_jobs < SATURATION_MIN_NEW_JOBS and ratio < SATURATION_MIN_NEW_RATIO:
            alerts.append(
                f'LOW YIELD: {new_jobs} new job(s) from {distinct} distinct listings '
                f'({ratio:.1%}) — below {SATURATION_MIN_NEW_RATIO:.0%}. The searches are '
                'returning jobs already processed; the query set or filters likely need widening'
            )
    return alerts


def attach_run_health(funnel: dict) -> list[str]:
    """Assess this run's discovery health, record it on the funnel, and log every alert.

    Called from BOTH exit paths. The zero-candidate path used to return before the main path ran
    this, so a run where every query errored reported "No alerts: page structure sound, regions
    distinct, yield acceptable" in audit section 1b and carried no NEEDS ATTENTION block in
    Telegram (2026-09-02) -- in the one branch where the alerts matter most. An alert that reaches
    neither the audit log nor the user is the most-repeated bug in this project.
    """
    alerts = assess_run_health(funnel)
    funnel['health_alerts'] = alerts
    for alert in alerts:
        logger.warning(f'Run health: {alert}')
    return alerts


def health_alert_block(alerts: list[str]) -> str:
    """The Telegram/console NEEDS ATTENTION section, or '' when the run was healthy."""
    if not alerts:
        return ''
    alert_lines = '\n'.join(f'  - {a}' for a in alerts)
    return f'⚠️ NEEDS ATTENTION ({len(alerts)}):\n{alert_lines}'


def recent_yield_history(limit: int = 5) -> list[str]:
    """Last few runs' yield, read back from cost_log.jsonl, to give an alert context.

    No new state file: every run already writes its funnel there.
    """
    path = RUN_DIR / 'cost_log.jsonl'
    if not path.exists():
        return []
    rows = []
    try:
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if 'funnel' in row:
                rows.append(row)
    except Exception as ex:
        logger.warning(f'could not read yield history from {path}: {ex}')
        return []

    out = []
    for row in rows[-limit:]:
        funnel = row.get('funnel') or {}
        seen = funnel.get('listings_distinct') or funnel.get('listings_seen') or 0
        new_jobs = (funnel.get('check_status') or {}).get('new', 0)
        out.append(f"{str(row.get('timestamp', ''))[:10]}: {new_jobs} new / {seen}")
    return out


def _anthropic_run_pass(client: ClaudeSDKClient, stage_stats: dict):
    """One Stage 1b request on the shared Claude Agent SDK session (the rollback path)."""
    async def run_pass(instruction: str) -> int | None:
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
    return run_pass


def _openrouter_run_pass(browser_call, tools: list[dict], stage_stats: dict, per_query: dict):
    """One Stage 1b request through the OpenRouter function-calling loop.

    A FRESH conversation per request. On the Anthropic path all six queries share one transcript,
    so context per turn grew 41K -> 162K across a run and every later query paid for every earlier
    snapshot. Here the browser session persists (the page stays where it was) while the transcript
    does not, so each query starts at the floor.
    """
    async def run_pass(instruction: str) -> int | None:
        query = tools_module._current_query or ''
        session = scrape_openrouter.ScrapeSession(browser_call, query)
        try:
            await session.run(build_scraper_prompt(), instruction, tools)
        finally:
            stage_stats['cost'] += session.cost
            stage_stats['input_tokens'] += session.usage['prompt'] - session.usage['cached']
            stage_stats['output_tokens'] += session.usage['completion']
            stage_stats['cache_read_input_tokens'] += session.usage['cached']
            entry = per_query.setdefault(query, {'cost': 0.0, 'iterations': 0, 'prompt': 0, 'cached': 0})
            entry['cost'] += session.cost
            entry['iterations'] += session.iterations
            entry['prompt'] += session.usage['prompt']
            entry['cached'] += session.usage['cached']
            logger.info(
                f'Stage 1b cost: "{query}" ${session.cost:.4f} over {session.iterations} iteration(s) '
                f'(prompt={session.usage["prompt"]:,} cached={session.usage["cached"]:,} '
                f'out={session.usage["completion"]:,})')
        return session.iterations
    return run_pass


async def _run_anthropic_scraper(playwright_mcp: dict, queries: list[str], stage_stats: dict) -> None:
    """The original Claude Agent SDK scraper, kept intact as the rollback path.

    Deliberately unchanged: it is what runs if the OpenRouter provider is unavailable, so it should
    stay the known-good implementation rather than drift alongside the new one.
    """
    options = ClaudeAgentOptions(
        setting_sources=[],
        strict_mcp_config=True,
        skills=[],
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
        model=ANTHROPIC_MODEL_NAME_LOW,
        max_turns=SCRAPER_MAX_TURNS_PER_QUERY,
    )
    async with ClaudeSDKClient(options) as scraper:
        await run_scraper(_anthropic_run_pass(scraper, stage_stats), queries, stage_stats)


async def run_scraper(run_pass, queries: list[str], stage_stats: dict) -> None:
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

    `run_pass(instruction) -> num_turns | None` is supplied by the caller, so everything in this
    function — the per-query loop, pacing, the region-verification check, the low-yield recovery
    pass — is shared between the OpenRouter and Anthropic scrapers rather than duplicated. The two
    differ only in how one request is executed.
    """
    for i, query in enumerate(queries, 1):
        console.print(f"[cyan]Stage 1b: query {i}/{len(queries)} — \"{query}\"[/cyan]")
        if i > 1:
            await _human_pause(SCRAPER_INTER_QUERY_DELAY_SECONDS, f'query {i}/{len(queries)}')
        tools_module._current_query = query
        before = _check_status_snapshot()
        before_distinct = _distinct_snapshot()
        try:
            turns = await run_pass(
                f'Search LinkedIn for this ONE query only: "{query}"\n\n'
                "Run every search listed in your instructions (one per region). Search on the "
                "query text alone, then set the location, date-posted and experience-level "
                "filters by CLICKING their chips — never by putting geoId/f_TPR in the URL. Then "
                "call run_ui_contract with the region name and do what it tells you, then "
                "harvest_listings, then record_listings. Clicking filter chips is fine; never "
                "click anything in the results LIST — the card and its Dismiss button are "
                "indistinguishable to you, and a stray click destroys a real job. If anything is "
                "missing or unreadable, call report_problem and stop rather than guessing. Do not "
                "ask for permission — call the tools directly. Do not navigate to individual job "
                "pages. Stop when you have worked through the searches for this query."
            )
        except ProviderUnavailableError as ex:
            # Systemic, not per-query: the provider is refusing every call, so the remaining
            # queries would each fail identically -- after paying their full human-emulation
            # pacing delay first. Abort the loop so the caller can fall back to the Anthropic
            # scraper, which is what the 2026-09-02 run should have done and could not, because
            # the handler below swallowed the 402 and let run_scraper return normally.
            logger.error(
                f'Stage 1b: provider unavailable on query "{query}" ({ex}) — aborting the query '
                f'loop after {i} of {len(queries)} so the caller can fall back'
            )
            # A query the loop never reached must not read as "searched, found nothing" in the
            # audit log; _queries_searched is what sections 2 and 3 report from.
            if partial := _check_status_delta(before):
                tools_module._check_status_per_query[query] = partial
            for pending in queries[i - 1:]:
                tools_module._queries_searched.setdefault(pending, 'error')
                tools_module._query_errors.setdefault(pending, str(ex))
            raise
        except Exception as ex:
            # One failed query must not abort the remaining ones.
            logger.warning(f'Stage 1b: query "{query}" failed: {ex}')
            tools_module._queries_searched[query] = 'error'
            tools_module._query_errors[query] = str(ex)
            # What it recorded before failing is real and already queued for Stage 2. Dropping it
            # left audit §3 reading "error | 0 listings | 2 queued" on 2026-09-10.
            if partial := _check_status_delta(before):
                tools_module._check_status_per_query[query] = partial
            continue

        # Every configured region must actually be searched, and each search must have been
        # verified. Both are prompted steps, and a model skips prompted steps: on the first live
        # run report_search was called 5 times across 12 expected searches, and one query searched
        # only one of its two regions -- with turn budget to spare (49 of 260), so this is choice,
        # not starvation. An unverified search is silently unfiltered, which is the whole failure
        # this guard exists to catch, so the omission itself has to be loud.
        expected_regions = [
            str(r.get('name') or r.get('linkedin_location', '')) for r in preferences.search_regions()
        ] or ['(unfiltered)']
        reported = {
            str(rec.get('region', '')) for rec in tools_module._search_reports
            if _same_query(rec.get('query', ''), query)
        }
        if missing := [r for r in expected_regions if r not in reported]:
            logger.warning(
                f'Stage 1b: query "{query}" never verified {len(missing)} of '
                f'{len(expected_regions)} region(s): {", ".join(missing)} — the search either did '
                'not run or ran without its filters confirmed.'
            )
            tools_module._ui_alerts.append({
                'kind': 'unverified_search', 'query': query, 'region': ', '.join(missing),
                'detail': f'no report_search for {", ".join(missing)} — search unrun or unverified',
            })

        delta = _check_status_delta(before)
        # DISTINCT, not the sum of call counts: two regions returning the same cards would
        # otherwise read as double coverage and push a collapsed query away from this retry.
        seen = len(_distinct_snapshot() - before_distinct)
        calls = sum(delta.values())
        if calls > seen:
            logger.info(
                f'Stage 1b: query "{query}" inspected {calls} listing(s) but only {seen} distinct '
                f'— {calls - seen} repeat check(s): the same listing recorded more than once, '
                f'across regions or re-harvested by a recovery pass. '
                f'See region_overlap for actual cross-region duplication.'
            )
        if seen < SCRAPER_MIN_LISTINGS_PER_QUERY:
            if seen == 0:
                logger.warning(
                    f'Stage 1b: query "{query}" inspected 0 distinct listings — retrying once '
                    'without the location filter (possible auth wall, block page, or empty shell).'
                )
                retry_instruction = (
                    f'That search surfaced no job listings at all for "{query}" — the results list was '
                    'empty or you hit a sign-in / verification wall.\n\n'
                    'If it was a CAPTCHA, a verification challenge, or an "unusual activity" notice: do '
                    'NOT retry. Say what you saw and stop — pushing through a block is the one thing '
                    'that can end this account.\n\n'
                    'Otherwise retry once: load '
                    f'https://www.linkedin.com/jobs/search-results/?keywords={quote_plus(query + ", remote")}&f_SAL= '
                    'and this time apply NO location filter at all — leave the location chip on '
                    'whatever it defaults to, and set only Date posted and Experience level. State '
                    'the result count you actually see, then walk the listings as instructed. If '
                    'you STILL see a wall or genuinely zero results, say so explicitly and stop.'
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
                    'return the identical list of jobs, the location filter is not applying — say '
                    'so explicitly rather than treating the query as exhausted.'
                )
            tools_module._ui_alerts.append({
                'kind': 'low_listings', 'query': query, 'region': '(all)',
                'detail': f'only {seen} distinct listing(s) on the first pass '
                          f'(below {SCRAPER_MIN_LISTINGS_PER_QUERY}) — recovery pass attempted',
            })
            await _human_pause(SCRAPER_INTER_SEARCH_DELAY_SECONDS, 'the recovery pass')
            try:
                await run_pass(retry_instruction)
            except Exception as ex:
                logger.warning(f'Stage 1b: recovery pass for "{query}" failed: {ex}')
            delta = _check_status_delta(before)
            # Recompute BOTH the same way the first pass did (line ~1406). Reassigning `seen` to a
            # call count here produced the impossible "100 distinct listing(s) of 50 checked", and
            # fed a call count into _queries_searched -- inflating apparent coverage for exactly
            # the queries that needed rescuing, which is the "saturated run looks healthy" shape.
            seen = len(_distinct_snapshot() - before_distinct)
            calls = sum(delta.values())

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
            tools_module._ui_alerts.append({
                'kind': 'stopped_early', 'query': query, 'region': '(all)',
                'detail': f'used only {turns} turns (expected >= {SCRAPER_MIN_TURNS_PER_QUERY}) '
                          '— stopped before completing a harvest cycle',
            })
        logger.info(
            f'Stage 1b: query "{query}" inspected {seen} distinct listing(s) of {calls} checked '
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
        setting_sources=[],
        strict_mcp_config=True,
        skills=[],
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
        model=ANTHROPIC_MODEL_NAME_LOW,
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
        setting_sources=[],
        strict_mcp_config=True,
        skills=[],
        model=ANTHROPIC_MODEL_NAME_LOW,
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
        'posting_language': (structured.get('posting_language') or '').strip().lower(),
        'relocation': structured.get('relocation', ''),
        'residency_scope': (structured.get('residency_scope') or '').strip().lower(),
        'workplace_type': (structured.get('workplace_type') or '').strip().lower(),
        'education_requirement': (structured.get('education_requirement') or '').strip().lower(),
    }
    logger.info(
        f"Extract fallback: {candidate['company']} — {candidate['title']}: "
        f"{len(snapshot)} snapshot chars → {len(extract['description'])} chars condensed"
    )
    return extract


_ONSITE_RE = re.compile(
    r'\bon[\s-]?site\b|\bin[\s-]?office\b|\bin[\s-]?person\b'
    # Phrasings that mean "you must be at the office" without using the usual words. All of these
    # returned '' before, which earned them the remote benefit of the doubt AND no commute cap.
    r'|\bon[\s-]?location\b|\boffice[\s-]?based\b|\bwork(?:ing)?\s+from\s+(?:our|the)\s+office'
    r'|\bpresence\s+in\s+the\s+office\b|\bbased\s+(?:out\s+)?of\s+(?:our|the)\s+\w+\s+office\b',
    re.IGNORECASE,
)
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


# Residency scope: does the posting pin you to the country it is anchored in, or offer a whole
# area? Judged from the JD's own words, read over `location` + `relocation` + the head of the
# description, because the wording lands in any of the three -- the extractors have always glued
# it into `location` as prose ('Romania (Remote within country)', 'Portugal (Remote - anywhere in
# the EU/Europe)', 'Netherlands (Remote; can be based in any EMEA Red Hat country)').

# Names of AREAS rather than countries. Every country-pinning pattern below requires an object
# that is NOT one of these, which is the whole trick: "must be based in Germany" pins you,
# "must be based in Europe" does not, and both are the same eight words up to the last one.
#
# WHAT BELONGS HERE, and it is not "every regional acronym". `area_wide` passes UNCONDITIONALLY,
# mirroring `broad_area`, whose justification is that "a posting offering a whole multi-country
# area is not limited to the countries it happens to name". That holds for an open-ended region
# and FAILS for a small closed bloc: "remote within DACH" is limited to exactly Germany, Austria
# and Switzerland, so it is closer to a country list than to an area, and listing it here would
# turn a role you must live in one of three specific countries for into an automatic pass.
# Closed blocs (DACH, Benelux, the Nordics, CEE, Iberia) are therefore deliberately absent and are
# judged on the anchor instead; `test_small_closed_blocs_are_judged_on_the_anchor` pins that.
#
# These patterns are matched case-INSENSITIVELY, unlike the place-name lists. That split is
# deliberate: this reads prose, where "remote within europe" is written every way imaginable,
# while the lists hold proper names, where 'Nice' the city is not 'nice' the adjective.
# Written as the names actually are -- 'EU', not 'eu' -- even though `re.IGNORECASE` on the
# patterns below means the case is not what does the matching. Folding is something you do at the
# point of comparison; a name stored folded can never be unfolded, and reading 'emea' in source
# gives no hint whether it is an acronym, a country or a typo.
_AREA_WORDS = (
    r'(?:EU|E\.U\.|EEA|EMEA|Europ(?:e|ean)|American?s?|North\s+America|South\s+America|'
    r'Schengen|LATAM|Latin\s+America|APAC|Asia[\s-]Pacific|MENA|'
    r'anywhere|world|globe|worldwide|country\s+where|countries\s+where)'
)
# What may sit between a preposition and an area name: one determiner and one adjective, so
# "within our EMEA region" and "in the wider Europe" still read as area wording. Deliberately only
# ONE filler word -- widen it and "based in Germany or Europe" starts reading as area-wide, where
# country_only is the conservative answer for a mixed statement.
#
# Shared by both patterns ON PURPOSE. The negative lookahead below and the area patterns must
# agree on what an area name looks like: when they drifted apart, "remote within our EMEA region"
# fell through BOTH -- the country branch stopped claiming it and no area branch picked it up.
_AREA_LEAD = r'(?:the\s+|an?\s+|our\s+|your\s+)?(?:\w+\s+)?'
_NOT_AREA = r'(?!' + _AREA_LEAD + _AREA_WORDS + r'\b)'

_AREA_WIDE_RE = re.compile(
    r'\b(?:work|working|remote|based|located|reside|residing|hire[sd]?|employed|eligible)\s+'
    r'(?:from|in|within|across|throughout|anywhere\s+in|to)\s+' + _AREA_LEAD + _AREA_WORDS + r'\b'
    r'|\banywhere\s+in\s+' + _AREA_LEAD + _AREA_WORDS + r'\b'
    r'|\b(?:work|remote|based|hire[sd]?|located)\s+(?:from\s+)?anywhere\b'
    r'|\bany\s+(?:\w+\s+){0,2}?(?:eu|eea|emea|european)(?:\s+\w+){0,2}?\s+country\b'
    r'|\b(?:emea|europe|eu)\s+(?:or|and)\s+(?:the\s+)?'
    r'(?:americas|north\s+america|south\s+america|eastern\s+us|us|usa|united\s+states)\b'
    r'|\b(?:eu|europe)\s*/\s*(?:eu|europe)\b'
    # Adjectival form, which has no preposition for the branches above to hang on:
    # "must be Europe-based", "EU-based candidates only".
    r'|\b' + _AREA_WORDS + r'[-\s]based\b'
    r'|\bremote\s*[-–—,;]\s*' + _AREA_LEAD + _AREA_WORDS + r'\b',
    re.IGNORECASE,
)

_COUNTRY_ONLY_RE = re.compile(
    r'\bremote\s+(?:only\s+)?(?:(?:with)?in|across|throughout)\s+(?:the\s+)?' + _NOT_AREA
    + r'(?:country|[a-zÀ-ɏ]+)\b'
    r'|\bmust\s+(?:be\s+)?(?:based|located|resident|reside|residing|live|living|work|working)'
    r'(?:\s+\w+){0,2}?\s+(?:in|from|within)\s+' + _NOT_AREA + r'(?:the\s+|an?\s+)?[a-zÀ-ɏ]'
    r'|\b(?:open\s+)?only\s+to\s+candidates\s+(?:residing|located|based)\s+in\s+'
    r'(?:the\s+|an?\s+)?' + _NOT_AREA + r'[a-zÀ-ɏ]'
    r'|\b(?:based|located|residing|resident)\s+in\s+' + _NOT_AREA + r'[a-zÀ-ɏ]+\s+only\b'
    r'|\bresidenc[ey]\s+in\s+' + _NOT_AREA + r'[a-zÀ-ɏ]+\s+(?:is\s+)?(?:required|mandatory)\b'
    r'|\bin[\s-]country\s+residenc[ey]\b'
    r'|\banywhere\s+(?:with)?in\s+(?:the\s+)?' + _NOT_AREA + r'[a-zÀ-ɏ]',
    re.IGNORECASE,
)

RESIDENCY_SCOPES = ('country_only', 'area_wide')


def derive_residency_scope(extract: dict) -> str:
    """'country_only' | 'area_wide' | '' -- where the posting says the holder must actually LIVE.

    This is not the anchor. "Romania (Remote)" anchors the role in Romania while the holder could
    live anywhere in the EU; "Romania (Remote within country)" does not. The geographic gate needs
    the second fact, and only the JD can supply it.

    Precedence is deliberately asymmetric, and the asymmetry is the OPPOSITE of `is_agency`'s
    (where the extractor's False is a judgement that wins): a `country_only` finding from EITHER
    the extractor or the text sticks, and it is checked before the area patterns. A wrong
    `area_wide` silently switches the geographic gate off for that job -- the exact failure this
    field exists to close -- while a wrong `country_only` surfaces as a rejection with a stated
    reason in the audit log and the run funnel. The union direction is the point: a restrictive
    finding from either source sticks, because only the permissive value can silently switch a gate
    off.

    The regexes are a fallback for when the extractor leaves the field unset; they carry the load
    only until the field is populated, which is why the patterns are anchored on a concrete
    non-area object rather than trying to parse the sentence.
    """
    explicit = str(extract.get('residency_scope') or '').strip().lower()
    if explicit not in RESIDENCY_SCOPES:
        explicit = ''

    haystack = (
        f"{extract.get('location', '')}\n{extract.get('relocation', '')}\n"
        f"{str(extract.get('description', ''))[:2000]}"
    )
    if explicit == 'country_only' or _COUNTRY_ONLY_RE.search(haystack):
        return 'country_only'
    if explicit == 'area_wide' or _AREA_WIDE_RE.search(haystack):
        return 'area_wide'
    return ''


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
    WARNING only. It never rejects a job and never changes a rating (see the Rating hard rules
    requirement in docs/requirements.md) — so a false positive is a stray bullet in a notification, not a lost job.
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
    if extract.get('posting_language'):
        lines.append(f"Posting written in: {extract['posting_language']}")
    if extract.get('implied_local_language'):
        lines.append(f"Implied local language: {extract['implied_local_language']}")
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
    # Folded HERE rather than in the accessor: the preference holds place names, and a name folded
    # on the way in cannot be unfolded on the way out. This rule is still a substring match, unlike
    # the two geographic lists -- see CLAUDE.md.
    location = extract['location'].casefold()
    needs_sponsorship = any(loc.casefold() in location for loc in sponsorship_locations)
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
    # Where the role is ANCHORED, and separately whether the JD demands a move. A bare location is
    # NOT a residency requirement — "Germany (Remote)" means the role sits in Germany and you could
    # live anywhere in the EU — so the two are judged apart, on different fields.
    if region := await location_rejection_reason(extract):
        return f'located in an excluded region: {region}'
    if relocation := str(extract.get('relocation') or '').strip():
        if region := await rejected_location(relocation):
            return f'relocation required to an excluded region: {region}'
    degree = derive_education_requirement(extract)
    if degree and degree in preferences.rejected_degrees():
        return f'requires advanced degree: {degree}'
    return None


def _unsupported_language(value: str) -> str:
    """The language named, if it is one the user does not work in; '' otherwise.

    Returns '' when the field is unset and when no languages are configured, so every rule built
    on this is inert under neutral defaults. Matching is by substring in the same direction as the
    `language_requirement` gate, so 'english (uk)' counts as english.
    """
    known = preferences.languages()
    text = str(value or '').strip().lower()
    if not known or not text:
        return ''
    return '' if any(lang in text for lang in known) else text


def foreign_posting_language(extract: dict) -> str:
    """The language the posting is WRITTEN IN, when the user does not read it; '' otherwise."""
    return _unsupported_language(extract.get('posting_language', ''))


def foreign_implied_local_language(extract: dict) -> str:
    """The language implied by the job's location, when it is not one the user speaks."""
    return _unsupported_language(extract.get('implied_local_language', ''))


async def derive_implied_local_language(extract: dict) -> str:
    """The working language IMPLIED by the job's location. Cosmetic: NO gate reads this.

    A place implies a language -- Italy implies Italian -- and that is world knowledge about a
    location, not a judgement about the posting. So the cached classifier is the only source:
    it cannot answer two Berlin jobs differently, within a run or across runs.

    The extractor used to supply this too, under a rule where its foreign finding won over the
    classifier's. That bought one thing -- a language named only in the body, like Valtech's
    "colleagues outside Quebec" on a `Canada (Remote)` posting -- at the cost of a fact about a
    place being decided per job by a model. A foreign language named in the body is no longer
    detected; a foreign language the JD is WRITTEN in still is, via `posting_language`, which is a
    fact about the page and stays a model judgement for that reason.
    """
    facts = await classify_location(extract.get('location', ''))
    return str(facts.get('implied_local_language') or '').strip().lower()


def commute_location_is_acceptable(location: str, place_names: Sequence[str] = ()) -> bool:
    """True if the user would travel to an OFFICE here. Drives the hybrid/on-site cap only.

    Deliberately NOT the same question as `location_is_allowed`. One list used to answer both, and
    because `Spain` was on it (somewhere the user would live), an on-site role in Ourense inherited
    the exemption and could score 5.

    `place_names` are the other names the classifier says this place goes by, resolved once per
    candidate into `extract['place_names']`; passing them is what lets a posting saying `Torino`
    match a list saying `Turin`.
    """
    candidates = (location, *place_names)
    return any(
        location_token_matches(token, candidate)
        for token in preferences.would_commute_here()
        for candidate in candidates
    )


def location_is_allowed(location: str, place_names: Sequence[str] = ()) -> bool:
    """True if the user would BE here — the geographic gate's exemption.

    Reads the union of `would_live_here` and `would_commute_here`: commuting somewhere implies
    being there, so a city on the commute list never needs repeating on the live list.
    """
    candidates = (location, *place_names)
    return any(
        location_token_matches(token, candidate)
        for token in preferences.would_live_here() + preferences.would_commute_here()
        for candidate in candidates
    )


# Countries this run guessed about, in order: country -> (guess, reason). Resolved at the end of
# the run into one appended block and one Telegram, rather than a call and a message per posting.
_location_guesses: dict[str, tuple[str, str]] = {}

_GUESS_PROMPT = """Someone filters job postings by where the role is anchored. These are the places
they have told us about:

- would live here: {live}
- would NOT live here: {not_live}

They have said nothing about "{country}". Judging ONLY by the pattern of the two lists above, which
is it more like?

Return ONLY a JSON object, no prose and no code fence:
{{"guess": "<would_live|would_not_live>", "reason": "<one short sentence>"}}
"""


def _trim(text: str, limit: int) -> str:
    """Cut at a word boundary. This string lands in a YAML comment and a Telegram message, and a
    mid-word cut ('...northern and western European countries l') reads as a bug in both."""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(' ', 1)[0].rstrip(',;:') + '…'


def queue_location_guess(country: str) -> None:
    """Note a country nothing on the lists names. Guessed once per run, resolved at run end."""
    if country and country not in _location_guesses:
        _location_guesses[country] = ('', '')


def reset_location_guesses() -> None:
    _location_guesses.clear()


async def resolve_location_guesses(stage_stats: dict | None = None) -> list[tuple[str, str, str]]:
    """Ask about each queued country, record the answers, return (place, guess, reason) rows.

    Fails open and quietly: no guess, nothing appended, nothing reported, every posting already
    passed. Same stance as the location classifier, for the same reason -- nothing the user wrote
    down matched, so an outage must not begin rejecting.
    """
    pending = [c for c, (guess, _r) in _location_guesses.items() if not guess]
    if not pending:
        return []
    live = list(preferences.would_live_here()) or ['(none)']
    not_live = list(preferences.would_not_live_here()) or ['(none)']
    for country in pending:
        try:
            content, cost = await chat_openrouter(
                _GUESS_PROMPT.format(live=live, not_live=not_live, country=country)
            )
            if stage_stats is not None:
                stage_stats['cost'] = stage_stats.get('cost', 0.0) + cost
            answer = extract_json_object(content)
            guess = str(answer.get('guess') or '').strip().lower()
            if guess not in {'would_live', 'would_not_live'}:
                raise ValueError(f'guess outside the vocabulary: {guess!r}')
            _location_guesses[country] = (
                'would live' if guess == 'would_live' else 'would NOT live',
                _trim(' '.join(str(answer.get('reason') or '').split()), 150),
            )
            logger.info(f'Location guess: {country} -> {_location_guesses[country][0]} (${cost:.6f})')
        except Exception as ex:
            logger.warning(f'Could not guess about {country!r}: {unwrap_exception(ex)}')
            _location_guesses.pop(country, None)
    return [(c, guess, reason) for c, (guess, reason) in _location_guesses.items() if guess]


def location_guess_notification(rows: list[tuple[str, str, str]]) -> str:
    """The batched Telegram. One message per run, not one per posting."""
    lines = [f'\u2753 Location guesses needing your confirmation ({len(rows)})', '']
    for place, guess, reason in rows:
        lines.append(f'\u2022 {place} \u2192 {guess}')
        if reason:
            lines.append(f'    {reason}')
    lines += [
        '',
        'These are GUESSES and gate nothing — those jobs were still rated and shown.',
        'Move each into would_live_here or would_not_live_here in preferences.yaml.',
    ]
    return '\n'.join(lines)


async def rejected_location(text: str, *, residency_spare: str = 'none') -> str:
    """The rejected region named in `text`, or '' when acceptable, unrecognised, or unconfigured.

    Purely geographic. This deliberately reads NO language field: an earlier design rejected on
    "non-English AND not on the acceptable list", which got the right answers for the wrong reason
    and would have excluded a French-language remote role in Canada — `acceptable_locations` lists
    Vancouver and British Columbia but not Canada itself. What language is spoken somewhere is a
    separate fact (`implied_local_language`), it warns only, and must never be folded back in here.

    Three tiers, cheapest first:
      1. unconfigured -> '' (no classifier call is ever made)
      2. an exempt location (`hybrid.acceptable_locations`) wins outright, so `france` can sit on
         the deny list while Toulouse and Nice still pass
      3. the deny list (`locations.exclude`)
      4. otherwise the cached classifier, and a rejection only when EVERY named country is in a
         rejected region — so 'the UK or the Netherlands' passes, and so does anything naming no
         country at all ('European Union', 'Remote (EMEA)')

    `residency_spare` says how far the JD lets the holder live from the anchor, and applies in
    tier 4 ONLY, beside `broad_area`. Deliberately not above: `locations.exclude` is the one
    mechanism no classifier variance can reach, and after this change it is the only way to drop
    a single EU country for remote roles, so it has to stay absolute.
      - 'none'     — judge the anchor (hybrid/on-site, or the JD pins residence to it)
      - 'any_area' — the JD offers a multi-country area; keep, exactly like `broad_area`
      - 'eu_only'  — the JD is silent; keep IF every named country is an EU member state, because
                     an EU anchor implies work rights the user has and a UK or Serbian one does not
    """
    # Whitespace only. The text is NOT lowercased any more: the lists hold proper names ('Málaga',
    # not 'malaga') and are matched against the text as written. `classify_location` is unaffected
    # -- `cache_key` lowercases for its own key, because case is not geography.
    haystack = ' '.join(str(text or '').split())
    if not haystack:
        return ''
    if not (preferences.would_live_here() or preferences.would_not_live_here()):
        return ''
    if location_is_allowed(haystack):
        return ''

    # Everything below needs to know EVERY country the text names, so it happens after the
    # (cached) classifier rather than against the raw string. Matching the deny list on the raw
    # text alone would reject "Germany or Spain (Remote)" on Germany while Spain is right there,
    # and "European Union (Remote, UK and EU)" on a country it only mentions in passing.
    facts = await classify_location(haystack)
    # Tier 3a: the same two lists again, now against every name the classifier says this place goes
    # by. This is what reaches an exonym no normalization can ('Sevilla' -> the listed 'Seville',
    # 'Torino' -> 'Turin'), and it is why the lists no longer need hand-maintained spelling pairs.
    # Order matches tiers 1 and 2 exactly: exempt wins over deny.
    place_names = facts.get('place_names') or []
    if any(location_is_allowed(name) for name in place_names):
        return ''
    # A whole multi-country area on offer is itself an unrejected option: "European Union (Remote,
    # UK and EU)" is not a UK-only role just because the UK is the one country it names. Found by
    # the live backtest, which flipped a 5/5 posting of exactly that shape. A single country phrased
    # expansively ("Berlin, Germany (Remote across Europe)") is NOT this — it stays anchored.
    if facts.get('broad_area'):
        return ''
    # The JD's own statement about residence, where broad_area is the location string's. Same
    # idea, two sources: the classifier reads the location, the extractor reads the body.
    if residency_spare == 'any_area':
        return ''
    countries_named = facts.get('countries') or []
    if residency_spare == 'eu_only' and countries_named and all(
        is_eu_member(country) for country in countries_named
    ):
        return ''
    deny_tokens = preferences.would_not_live_here()

    if not countries_named:
        # The classifier resolved no country, so there is nothing to reason about per-country --
        # but the text may still literally name a place on the list ('Blockedland (Remote)').
        return next(
            (token for token in deny_tokens
             if location_token_matches(token, haystack)
             or any(location_token_matches(token, name) for name in place_names)),
            '',
        )

    def _denied(country: str) -> str:
        # Against the country AND, when only one is named, the other names it goes by -- so a
        # posting saying 'Praha, Czech Republic' is denied by a list saying 'Czechia'.
        names = [country, *(place_names if len(countries_named) == 1 else ())]
        return next(
            (token for token in deny_tokens
             if any(location_token_matches(token, name) for name in names)), '',
        )

    # Rejected only when EVERY country on offer is one you would not live in -- "Spain or the
    # Netherlands" leaves you somewhere to be, and so does a whole area. Same rule the region
    # policy used before the lists replaced it.
    denials = [_denied(country) for country in countries_named]
    if all(denials):
        return denials[0]

    undecided = [
        country for country, denial in zip(countries_named, denials)
        if not denial
        and not any(location_token_matches(t, country) for t in preferences.would_live_here())
        and not any(location_token_matches(t, country) for t in preferences.not_yet_bucketed())
    ]
    # Nothing the user wrote down names these. Guess once, record it, tell them -- and KEEP the
    # posting. A guess must never reject: the point of the bucketing loop is that the user
    # corrects it, which they cannot do for a job they were never shown.
    for country in undecided:
        queue_location_guess(country)
    return ''


def residency_spare(extract: dict) -> str:
    """'none' | 'eu_only' | 'any_area' — how far the JD lets the holder live from the anchor."""
    if derive_workplace_type(extract) in {'hybrid', 'onsite'}:
        return 'none'          # an office pins you regardless of what the text says
    return {'country_only': 'none', 'area_wide': 'any_area'}.get(derive_residency_scope(extract), 'eu_only')


async def location_rejection_reason(extract: dict) -> str:
    """The geographic reason to reject this posting, or '' to keep it.

    `rejected_location` asks "would the user live in the place this text names". That is the right
    question only once you know the posting actually requires living there. A remote role ANCHORED
    in Romania may mean "live in Romania" or "live anywhere in the EU", and those are opposite
    answers to the same location string — which is why the anchor alone was the wrong input.

    The `relocation` hard rule is unchanged and still runs separately, unspared: it is the JD
    asserting a categorical requirement rather than code inferring one from a location string.
    """
    spare = residency_spare(extract)
    reason = await rejected_location(extract.get('location', ''), residency_spare=spare)
    if not reason or spare != 'none' or derive_residency_scope(extract) != 'country_only':
        return reason
    # No 'relocation required' and no 'language' in this suffix: _hard_rule_category is an ordered
    # substring dispatcher and either phrase would misbucket the funnel counter.
    return f'{reason} — residency pinned to the anchor country'


def apply_rating_caps(extract: dict, rating: int) -> tuple[int, str]:
    """Deterministic post-rating ceilings. Returns (rating, reason) — reason is '' if uncapped.

    A cap is not a rejection: the job is still saved and still appears in the audit log, it just
    never crosses the >=4 notification threshold. This backstops the evaluator prompt, which has
    demonstrably rated a hybrid role in an unacceptable location 4/5 while naming the hybrid
    location as a drawback in its own reasoning — and, on the Valtech posting, rated a
    French-language JD 4/5 after silently translating it into English while condensing.

    Several caps can apply at once; the lowest wins and every applicable reason is reported, so
    the log line and the saved job say everything that held the rating down.
    """
    caps: list[tuple[int, str]] = []

    workplace_type = derive_workplace_type(extract)
    if workplace_type in {'hybrid', 'onsite'} and not commute_location_is_acceptable(
        extract.get('location', ''), extract.get('place_names') or ()
    ):
        location = extract.get('location') or 'unspecified location'
        caps.append((
            preferences.hybrid_rating_cap(),
            f'{workplace_type} in {location} (not an acceptable hybrid location)',
        ))

    if language := foreign_posting_language(extract):
        caps.append((
            preferences.foreign_language_rating_cap(),
            f'posting written in {language}',
        ))

    applicable = [(cap, reason) for cap, reason in caps if rating > cap]
    if not applicable:
        return rating, ''
    return min(cap for cap, _ in applicable), '; '.join(reason for _, reason in applicable)


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
        if not commute_location_is_acceptable(extract.get('location', ''), extract.get('place_names') or ()):
            warning += ' (not somewhere you would commute)'
        warnings.append(warning)

    # Two independent language facts. A posting written in another language gets the first (and a
    # rating cap); an English posting in a non-English workplace gets only the second. The Valtech
    # posting — a French JD the extractor handed back as English prose — was rated 4/5 and
    # notified with neither, which is why neither is left to the rater.
    if language := foreign_posting_language(extract):
        warnings.append(f'Posting written in {language.title()} — not English')

    if implied_local_language := foreign_implied_local_language(extract):
        location = extract.get('location') or 'location not stated'
        warnings.append(f'Implied local language: {implied_local_language.title()} — {location}')

    location = extract.get('location') or ''
    if guessed := next(
        (token for token in preferences.not_yet_bucketed()
         if location_token_matches(token, location)
         or any(location_token_matches(token, n) for n in extract.get('place_names') or ())),
        '',
    ):
        warnings.append(
            f'Location not yet bucketed: {guessed} — the agent guessed about it; confirm it in '
            f'preferences.yaml'
        )

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


async def _rate_with_anthropic(evaluator_prompt: str, extract_text: str, stage_stats: dict,
                               model: str = ANTHROPIC_MODEL_NAME_MEDIUM) -> dict:
    options = ClaudeAgentOptions(
        setting_sources=[],
        strict_mcp_config=True,
        skills=[],
        model=model,
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
    """Stage 2d: one non-agentic rating call; MODEL_NAME_RATING's family prefix picks the
    route (route_for) and the model together."""
    route = route_for(MODEL_NAME_RATING)
    if route == 'openrouter':
        result, cost_usd = await rate_with_openrouter(
            evaluator_prompt, f'Evaluate this job posting:\n\n{extract_text}', model=MODEL_NAME_RATING)
        stage_stats['cost'] += cost_usd
        return result
    if route == 'ollama':
        return await rate_with_ollama(
            evaluator_prompt, f'Evaluate this job posting:\n\n{extract_text}', model=MODEL_NAME_RATING)
    return await _rate_with_anthropic(evaluator_prompt, extract_text, stage_stats, model=MODEL_NAME_RATING)


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
    # Before the location branch: the relocation reason contains the word "location" too, and
    # this dispatcher is ordered substring matching — it must not depend on that accident.
    if 'relocation required' in reason:
        return 'hard_ruled_relocation'
    if 'excluded region' in reason:
        return 'hard_ruled_location'
    if 'language' in reason:
        return 'hard_ruled_language'
    if 'degree' in reason:
        return 'hard_ruled_education'
    return 'hard_ruled_other'


async def evaluate_all_candidates(
    candidates: list[dict], playwright_mcp: dict, evaluator_prompt: str,
    profile_block: str, stage_stats: dict, funnel: dict | None = None, audit: bool = False,
    triage_enabled: bool = True,
) -> None:
    """Stage 2: per candidate — Haiku extract, deterministic hard rules, local triage,
    then one configurable-model rating call. All sharing one browser.

    When ``audit`` is set, gate-killed candidates (hard-ruled / triaged-out) are STILL
    sent through the strong rater so we can detect false negatives (gate dropped it but
    the strong model rates it >=3). Normal saving/notification behaviour is unchanged.
    ``funnel`` (if provided) accumulates per-stage drop counts for the run summary.

    ``triage_enabled`` is the run-level switch the local-model preflight turns off: a missing
    OLLAMA_MODEL_NAME_TRIAGE is reported ONCE and triage is skipped, rather than every job
    re-discovering the same misconfiguration and logging an identical warning (2026-08-25, 63 of
    them in one run).
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
            if route_for(MODEL_NAME_EXTRACTOR) == 'openrouter':
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
            # Resolve the implied local language ONCE, here, and write it back — so
            # format_extract_text and build_deterministic_warnings (both sync) keep reading a plain
            # field. Cosmetic only: no gate reads it, by design.
            extract['implied_local_language'] = await derive_implied_local_language(extract)
            # Same treatment for residency_scope, and for the same reason: resolve once here so
            # the sync consumers (the gate, the warnings) read a plain field, and log raw->derived
            # so a fallback that silently overrides the extractor stays countable.
            raw_residency_scope = str(extract.get('residency_scope') or '')
            extract['residency_scope'] = derive_residency_scope(extract)
            # Every other name this place goes by, resolved ONCE (the classifier answer is already
            # cached from derive_implied_local_language above, which now calls it unconditionally,
            # so this is always free) and written back, so the sync consumers below read a plain
            # field instead of each needing to be async.
            extract['place_names'] = (
                await classify_location(extract.get('location', ''))
            ).get('place_names') or []
            extract_text = format_extract_text(candidate, extract)
            logger.info(
                f"Extract signal: {candidate['company']} — {candidate['title']}: "
                f"date_posted={extract.get('date_posted')!r} location={extract.get('location')!r} "
                f"closed={extract.get('closed')} language_requirement={extract.get('language_requirement')!r} "
                f"posting_language={extract.get('posting_language')!r} "
                f"implied_local_language={extract.get('implied_local_language')!r} "
                f"relocation={extract.get('relocation')!r} "
                f"residency_scope={raw_residency_scope!r}->{extract.get('residency_scope')!r} "
                f"place_names={extract.get('place_names')} "
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
            if TRIAGE_ENABLED and triage_enabled:
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
                f"{rating}/5 via {MODEL_NAME_RATING}{triage_note} — {result['reasoning']}"
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
        setting_sources=[],
        strict_mcp_config=True,
        skills=[],
        tools=[],
        system_prompt=evaluator_prompt,
        permission_mode='bypassPermissions',
        model=ANTHROPIC_MODEL_NAME_HIGH,
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


# Alerts raised before run_non_interactive() starts. It resets tools_module._ui_alerts to [] on
# entry, so anything appended there during startup would be silently discarded — a health signal
# that reaches nothing is this project's most-repeated bug, not a new one.
_startup_ui_alerts: list[dict] = []


def _parse_semver(version: str) -> tuple[int, ...] | None:
    """('0.0.80') -> (0, 0, 80); None for anything not purely numeric dotted parts."""
    parts = version.strip().split('.')
    if not all(part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def check_playwright_mcp_version() -> str | None:
    """Return the published @playwright/mcp version if it is NEWER than the pin, else None.

    Fails open on absolutely everything — offline, timeout, a registry outage, a body that does
    not parse. This runs before the run lock is taken and before the browser starts, so a network
    hiccup here must cost a log line, never a run.
    """
    try:
        response = requests.get(
            PLAYWRIGHT_MCP_REGISTRY_URL, timeout=PLAYWRIGHT_MCP_VERSION_CHECK_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        published = str(response.json()['version'])
    except Exception as ex:
        logger.info(
            f'Could not check for a newer @playwright/mcp than the pinned '
            f'{PLAYWRIGHT_MCP_VERSION} at {PLAYWRIGHT_MCP_REGISTRY_URL} — continuing on the pin: '
            f'{type(ex).__name__}: {ex}'
        )
        return None

    pinned_parts, published_parts = _parse_semver(PLAYWRIGHT_MCP_VERSION), _parse_semver(published)
    if pinned_parts is None or published_parts is None:
        logger.info(
            f'Cannot compare @playwright/mcp versions (pinned {PLAYWRIGHT_MCP_VERSION!r}, '
            f'published {published!r}) — continuing on the pin'
        )
        return None
    return published if published_parts > pinned_parts else None


def _playwright_upgrade_instructions(published: str) -> str:
    return (
        f'  1. npx --yes @playwright/mcp@{published} --version\n'
        f"  2. set PLAYWRIGHT_MCP_VERSION = '{published}' in src/agentic_job_search/config.py\n"
        f'  3. re-run and watch one live search — this drives the real LinkedIn account'
    )


def prompt_playwright_mcp_upgrade(interactive: bool) -> None:
    """Offer the choice BEFORE the run lock, the job load, or npx — so exiting costs nothing.

    Asks ONLY in interactive mode with a real TTY. Both conditions are load-bearing, and gating on
    the TTY alone was a bug: `-n` is the autonomous mode, run by cron *and* by hand from a terminal,
    where isatty() is True — so a hand-run `-n` stopped dead on a question, which is precisely the
    failure this whole mechanism exists to prevent. `-n` means "do not ask me things".

    Anywhere it does not ask, it warns, raises a health alert, and continues on the pin.
    """
    published = check_playwright_mcp_version()
    if not published:
        return

    detail = f'@playwright/mcp {published} is available; this run uses the pinned {PLAYWRIGHT_MCP_VERSION}'
    if not interactive or not sys.stdin.isatty():
        logger.warning(f'{detail}. To upgrade:\n{_playwright_upgrade_instructions(published)}')
        _startup_ui_alerts.append({
            'kind': 'playwright_mcp_outdated', 'query': '(all)', 'region': '(all)',
            'detail': detail})
        return

    console.print(f'\n[yellow]{detail}.[/yellow]')
    console.print('[dim]To upgrade:[/dim]')
    console.print(f'[dim]{_playwright_upgrade_instructions(published)}[/dim]')
    try:
        choice = input(
            f'\n[u] exit so you can upgrade   [c] continue on {PLAYWRIGHT_MCP_VERSION} (default): '
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        # Ctrl-D, Ctrl-C, or a TTY that reports itself interactive but never delivers a line.
        console.print(f'\n[dim]No answer read — continuing on pinned {PLAYWRIGHT_MCP_VERSION}.[/dim]')
        logger.warning(f'{detail}; no answer read, continuing on the pin')
        return
    if choice.startswith('u'):
        console.print('\n[bold]Exiting without running.[/bold] Do the three steps above, then re-run.')
        sys.exit(0)
    logger.info(f'Continuing on pinned @playwright/mcp {PLAYWRIGHT_MCP_VERSION} ({published} available)')


async def start_playwright_server(port: int, browser_mode: str = 'minimized') -> asyncio.subprocess.Process:
    cmd = [
        'npx', '--yes', PLAYWRIGHT_MCP_PACKAGE,
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
    # `npx --yes <pinned package> --help`). Harmless noise — the URL below is what we
    # actually use, not the banner's copy-paste config.
    proc = await asyncio.create_subprocess_exec(*cmd)
    try:
        console.print(f'[dim]→ GET http://localhost:{port}/mcp (polling until ready)[/dim]')
        for _ in range(30):
            await asyncio.sleep(1)
            # A dead npx polls exactly like a slow one, so without this a bad pin costs 30s of
            # silence and then surfaces as a connection error naming neither npx nor the version.
            if proc.returncode is not None:
                raise RuntimeError(
                    f'{PLAYWRIGHT_MCP_PACKAGE} failed to launch: npx exited '
                    f'{proc.returncode} (its output is above). If that version does not exist or '
                    f'was yanked, correct PLAYWRIGHT_MCP_VERSION in config.py.'
                )
            try:
                requests.get(f'http://localhost:{port}/mcp', timeout=1)
                break
            except Exception:
                pass
    finally:
        if tmp_config:
            Path(tmp_config).unlink(missing_ok=True)
    return proc


async def run_stage_1b(port: int, playwright_mcp: dict, queries: list[str], scraping_stats: dict) -> bool:
    """Run Stage 1b, falling back to the Anthropic scraper on failure.

    The route (and model) is MODEL_NAME_SCRAPER: an OPENROUTER_* family value runs the
    function-calling loop; anything else routes Stage 1b straight to the Anthropic scraper.

    Extracted from `run_non_interactive` so the fallback is REACHABLE BY A TEST. It was previously
    inline, and the bug it hides is not hypothetical: on 2026-09-02 an OpenRouter 402 was swallowed
    by `run_scraper`'s per-query handler, so this `except` never saw it, `scraped` stayed True, and
    the Anthropic scraper -- sitting right there, correct, and never called -- did not run. Nothing
    could have caught that, because nothing could call this code without a live browser.

    Returns True if the OpenRouter path completed, False if the Anthropic fallback ran.
    """
    if route_for(MODEL_NAME_SCRAPER) == 'openrouter':
        # Browser tools come from the running Playwright server's own schemas rather than a
        # hand-transcribed constant, so a server-side change cannot drift silently.
        try:
            async with mcp_session(f'http://localhost:{port}/mcp') as browser_call:
                tool_defs = (scrape_openrouter.browser_tool_defs(await browser_call.list_tools())
                             + scrape_openrouter.LOCAL_TOOL_DEFS)
                logger.info(f'Stage 1b: {len(tool_defs)} tools exposed to '
                            f'{MODEL_NAME_SCRAPER}')
                await run_scraper(
                    _openrouter_run_pass(browser_call, tool_defs, scraping_stats,
                                         tools_module._scrape_per_query),
                    queries, scraping_stats)
            return True
        except Exception as ex:
            # A provider outage must not end the run: fall through to the Anthropic scraper,
            # exactly as query generation, rating and extraction already do.
            logger.warning(f'Stage 1b: OpenRouter scraper failed ({unwrap_exception(ex)}) — '
                           'falling back to the Anthropic scraper')
            tools_module._ui_alerts.append({
                'kind': 'provider_fallback', 'query': '(all)', 'region': '(all)',
                'detail': f'OpenRouter scraper failed, fell back to Anthropic: {unwrap_exception(ex)}'})

    await _run_anthropic_scraper(playwright_mcp, queries, scraping_stats)
    return False


async def run_non_interactive(browser_mode: str = 'headless', audit: bool = False, audit_opus: int = 0) -> None:
    tools_module._candidates = []
    tools_module._candidates_per_query = {}
    tools_module._job_extracts = []
    tools_module._check_status_counts = {}
    tools_module._listing_records = {}
    tools_module._queries_searched = {}
    tools_module._query_errors = {}
    tools_module._check_status_per_query = {}
    tools_module._queue_skipped_counts = {}
    tools_module._current_query = None
    tools_module._current_region = None
    tools_module._distinct_listing_ids = set()
    tools_module._search_ids = {}
    tools_module._search_reports = []
    tools_module._ui_alerts = list(_startup_ui_alerts)
    location.reset_countries_seen()
    reset_location_guesses()
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
        "location_review": new_stage_stats(),
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
        await run_stage_1b(port, playwright_mcp, queries, stage_stats["scraping"])

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
            # Health is assessed HERE too, not only on the main path below: this branch returns
            # before that code, so without this a run whose every query failed reported itself as
            # healthy in both the audit log and Telegram.
            # Advisory review of the country lists. Must run BEFORE the funnel snapshots
            # `ui_alerts`, and on BOTH exit paths -- two copies of an alert-raising step is
            # exactly how one of them came to be missing before.
            if guess_rows := await resolve_location_guesses(stage_stats.get('location_review')):
                preferences.append_not_yet_bucketed(guess_rows)
                await _send_pipeline_notification(location_guess_notification(guess_rows))
                tools_module._ui_alerts.append({
                    'kind': 'location_recommendations', 'query': '', 'region': '',
                    'detail': f'{len(guess_rows)} location guess(es) appended to not_yet_bucketed: '
                              + ', '.join(f'{p} -> {g}' for p, g, _r in guess_rows),
                })

            if detail := await location_review.review_location_lists(stage_stats.get('location_review')):
                tools_module._ui_alerts.append(
                    {'kind': 'location_recommendations', 'query': '', 'region': '', 'detail': detail}
                )
            zero_funnel = {
                "queries_generated": len(queries),
                "listings_seen": sum(check_status_counts.values()),
                "listings_distinct": _listings_distinct(),
                "region_overlap": tools_module.region_overlap_report(),
                "ui_alerts": list(tools_module._ui_alerts),
                "check_status": check_status_counts,
                "check_status_per_query": dict(tools_module._check_status_per_query),
                "queries_failed": dict(tools_module._query_errors),
                "queue_skipped": dict(tools_module._queue_skipped_counts),
                "candidates_queued": 0,
            }
            health_alerts = attach_run_health(zero_funnel)
            fail_lines = [
                "Job search run FAILED",
                "• Error: 0 candidates found",
                f"• Queries ({len(queries)}):\n{query_lines}",
                f"• Listings seen: {sum(check_status_counts.values())} {check_status_counts}",
                f"• Elapsed: {elapsed_mins:.1f} min",
                f"• Total cost: ${total_cost:.4f}",
            ]
            if block := health_alert_block(health_alerts):
                fail_lines.insert(1, block)
                if history := recent_yield_history():
                    fail_lines.append("• Recent yield:\n" + "\n".join(f"  - {h}" for h in history))
            await _send_pipeline_notification("\n".join(fail_lines))
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
                funnel=zero_funnel,
            )
            console.print(f"[dim]Run audit log: {audit_log_path}[/dim]")
            return

        # Stage 2: all evaluations share the same browser via the SSE server
        console.print("[yellow]Stage 2: Evaluating candidates ...[/yellow]\n")
        reference_block = await build_reference_summary(stage_stats["reference_summary"])
        evaluator_prompt = build_evaluator_prompt(reference_block)
        profile_block = build_profile_block(reference_block)

        # One check, before the per-job loop: an OLLAMA_MODEL_NAME_TRIAGE that is not installed
        # silently disables the free triage gate on every candidate, and the run still looks normal.
        triage_enabled = True
        if TRIAGE_ENABLED:
            preflight_problem = await preflight_local_model()
            if preflight_problem:
                triage_enabled = False
                logger.error(f'Local triage disabled for this run: {preflight_problem}')
                tools_module._ui_alerts.append({
                    'kind': 'local_model_missing', 'query': '(all)', 'region': '(all)',
                    'detail': preflight_problem})

        await evaluate_all_candidates(
            candidates, playwright_mcp, evaluator_prompt, profile_block, stage_stats,
            funnel=funnel, audit=audit, triage_enabled=triage_enabled,
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
    if guess_rows := await resolve_location_guesses(stage_stats.get('location_review')):
        preferences.append_not_yet_bucketed(guess_rows)
        await _send_pipeline_notification(location_guess_notification(guess_rows))
        tools_module._ui_alerts.append({
            'kind': 'location_recommendations', 'query': '', 'region': '',
            'detail': f'{len(guess_rows)} location guess(es) appended to not_yet_bucketed: '
                      + ', '.join(f'{p} -> {g}' for p, g, _r in guess_rows),
        })

    if detail := await location_review.review_location_lists(stage_stats.get('location_review')):
        tools_module._ui_alerts.append(
            {'kind': 'location_recommendations', 'query': '', 'region': '', 'detail': detail}
        )

    funnel_summary = {
        "queries_generated": len(queries),
        "listings_seen": sum(check_status_counts.values()),
        "listings_distinct": _listings_distinct(),
        "region_overlap": tools_module.region_overlap_report(),
        "ui_alerts": list(tools_module._ui_alerts),
        "check_status": check_status_counts,
        "check_status_per_query": dict(tools_module._check_status_per_query),
        "queries_failed": dict(tools_module._query_errors),
        "queue_skipped": dict(tools_module._queue_skipped_counts),
        "candidates_queued": len(candidates),
        "scrape_per_query": dict(tools_module._scrape_per_query),
        **funnel,
    }
    logger.info(f"Run funnel: {json.dumps(funnel_summary)}")

    # Discovery-health alerts. A run that finds nothing because the page changed, because a region
    # collapsed, or because the pool is exhausted must not look the same as a healthy quiet run.
    health_alerts = attach_run_health(funnel_summary)

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
    if block := health_alert_block(health_alerts):
        stats_lines.insert(1, block)
        if history := recent_yield_history():
            stats_lines.append("• Recent yield:\n" + "\n".join(f"  - {h}" for h in history))
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
    # ratings — everything logged via the logging module lands in both). Written to
    # run_dir/logs/run-<YYYY-MM-DD>_<HHMMSS>.log via the shared logging setup.
    setup_logging('run', RUN_DIR, level=os.environ.get('LOG_LEVEL', 'INFO'))
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
        "--no-version-check",
        action="store_true",
        help="Skip the startup check for a newer @playwright/mcp than the pinned version",
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

    # Before the run lock, the processed-job load, and any npx spawn: choosing to upgrade here
    # costs nothing to unwind. The browser server does not start until run_non_interactive (or,
    # interactively, until ClaudeSDKClient spawns the stdio server) far below.
    if not args.no_version_check:
        prompt_playwright_mcp_upgrade(interactive=interactive)

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
        await tools_module.categorize_save_dir_pdfs()
        await tools_module.ingest_save_dir_applied_pdfs()
        await tools_module.load_applied_jobs()

        if interactive:
            options = ClaudeAgentOptions(
                setting_sources=[],
                strict_mcp_config=True,
                skills=[],
                system_prompt=build_system_prompt(interactive=True),
                mcp_servers={
                    "playwright": {
                        "type": "stdio",
                        "command": "npx",
                        "args": [
                            "--yes",
                            PLAYWRIGHT_MCP_PACKAGE,
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
