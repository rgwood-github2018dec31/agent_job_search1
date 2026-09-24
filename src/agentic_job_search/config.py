from utils_tools_n_agents_common.models import (
    OPENROUTER_MODEL_NAME_DEFAULT_AGENTIC,
    OPENROUTER_MODEL_NAME_DEFAULT_INTELLIGENCE,
    OPENROUTER_MODEL_NAME_SCRAPER,
)

# Job search parameters
JOB_MAX_AGE_DAYS = 21
JOB_STALE_AGE_DAYS = 30  # hard rule: postings older than this are auto-rated RATING_AUTO_REJECT
MAX_REFERENCE_JOBS = 20
THINKING_MAX_CHARS = 1000
# Applied-job corpus: only records applied to within this window feed query generation,
# the ideal-role profile, and the already-applied company blocklist. Older PDFs are kept
# on disk but never read. An applied-to company must stay blocked long enough to cover a
# repost cycle, so this matches COMPANY_BLACKLIST_EXPIRY_DAYS and both lapse rules read
# alike. Was 90 until 2026-09-15: a GitLab role applied to on 2026-05-12 aged out of the
# blocklist on 08-10, and the identical reposted role — same title, same location, a new
# LinkedIn job id — was notified as a 5 on 09-13, 124 days after the application.
APPLIED_JOBS_HORIZON_DAYS = 180
# Company blacklist: an entry stops rejecting this many days after its `added` date. A standing
# "never show me this company" decision goes stale — six months on, the reason for it may not
# hold any more — so the entry lapses and is reported rather than silently outliving its reason.
COMPANY_BLACKLIST_EXPIRY_DAYS = 180
# Recruiter reposts: how far back to look for an already-notified posting from the SAME agency
# before deciding a new one is the same role. Agencies re-advertise one role under a new LinkedIn
# job id every few days with a different country and day rate — six notifications for one pharma
# role between 2026-09-02 and 09-10 — and each new id defeats (site, job_id) deduplication.
RECRUITER_REPOST_WINDOW_DAYS = 14
# ...and how many of those postings the judgement call is shown, newest first. An agency that posts
# daily would otherwise grow the prompt without bound: every prior carries a description capped at
# RECRUITER_DESCRIPTION_MAX_CHARS (~9.7K chars at 7 priors was measured before that cap was raised
# on 2026-09-24; the ceiling is now this many priors times that cap, still far inside any model's
# context). A repost
# is re-advertised within days of the original, so the newest few are the ones that can match.
RECRUITER_REPOST_MAX_PRIORS = 8

# Job-fit rating scale (1-5). The threshold and rejection rating are structural facts decided
# in code, not model discretion points (docs/requirements.md, "A structural fact is decided
# in code") — the predicates below are the only way call sites touch them.
RATING_MIN = 1
RATING_MAX = 5
# Jobs at/above this rating reach the user (Telegram notification, high-rated count, recruiter-
# repost checks, Opus-audit FALSE NEGATIVE verdicts). If this ever moves off 4, update the
# "rated 4 or 5" prose in CLAUDE.md and docs/requirements.md to match.
RATING_NOTIFICATION_THRESHOLD = 4
# The rating recorded for every deterministic rejection (hard rules, already-applied): the job
# is saved and audited, never notified.
RATING_AUTO_REJECT = RATING_MIN
# Surfaced but not acted on — the --audit-opus mid_rated pool. Derived from the two ratings it
# sits between so it can never drift out of step with them.
RATINGS_MID_RATED = frozenset(range(RATING_AUTO_REJECT + 1, RATING_NOTIFICATION_THRESHOLD))


def should_notify_based_on_rating(rating: int) -> bool:
    """True when a job at/above RATING_NOTIFICATION_THRESHOLD reaches the user."""
    return rating >= RATING_NOTIFICATION_THRESHOLD


def is_auto_reject_rating(rating: int) -> bool:
    """True for the deterministic-rejection rating (hard rules, already-applied)."""
    return rating == RATING_AUTO_REJECT


def is_mid_rated(rating: int) -> bool:
    """True when the job surfaced but was not acted on — the --audit-opus mid_rated pool."""
    return rating in RATINGS_MID_RATED


# NOTE: personal preferences — search regions, acceptable hybrid locations, the hybrid rating cap,
# work-authorization/language/education gates, and target titles — deliberately do NOT live here.
# They are loaded from the gitignored run_dir/preferences.yaml via preferences.py, so this repo
# carries no information about whoever is running it. See preferences.example.yaml.

# Local/remote LLM MCP tool servers (started via their scripts/start-tool-server.sh)
LLM_LOCAL_MCP_URL = 'http://127.0.0.1:8002/mcp'
LLM_OPENROUTER_MCP_URL = 'http://127.0.0.1:8006/mcp'
# Wall-clock cap for one LLM call (chat/generate) through the shared MCP client. Its own default
# (30s) is sized for quick sends like Telegram, and silently applied to every LLM call when this
# repo moved onto it: a Stage 1b scraper round resending a conversation full of 30k-char snapshots
# regularly takes longer, and 2026-09-14 lost 4 of 6 queries plus most of Stage 2 to
# "timed out after 30.0s". 300s matches the SSE read bound the previous in-repo client had.
LLM_MCP_CALL_TIMEOUT_SECONDS = 300.0
# Machine-local Ollama tag for Stage 2c triage (one cheap 1-5 JSON score, free and fast —
# deliberately NOT sized to a 20-35B tier): granite4.1:3b measures 65.9 tok/s / 3.9s cold start /
# 2.1GB on this machine and advertises structured JSON output as a first-class capability.
#
# This is a MACHINE-LOCAL Ollama tag and it can vanish without warning: `qwen3.6:latest` sat here
# until a re-pull replaced it with `qwen3.6:27b-mlx`/`35b-mlx`, and Stage 2c triage then failed open
# on every job for ten days while the run log said only "unhandled errors in a TaskGroup". That is
# what preflight_local_model() in triage.py now catches, once per run, naming what IS installed.
# The OLLAMA_ prefix is the one place that says "machine-local tag" — unlike the OpenRouter-routed
# consts from utils_tools_n_agents_common.models, this value is defined here and nowhere else.
OLLAMA_MODEL_NAME_TRIAGE = 'granite4.1:3b'
# Single-call OpenRouter tasks (query generation, company matching, rating) and agentic loops
# (extraction, scraping) each name their shared const DIRECTLY at call sites — no local aliases:
# query-gen / company-match / rating import OPENROUTER_MODEL_NAME_DEFAULT_INTELLIGENCE, extraction
# imports OPENROUTER_MODEL_NAME_DEFAULT_AGENTIC (flash tier for cost, not the strongest single-call
# model), scraping imports OPENROUTER_MODEL_NAME_SCRAPER. All come from
# utils_tools_n_agents_common.models — a model swap is one edit there, not a per-project hunt.

# Stage 1 (discovery) configuration
MAX_SEARCH_QUERIES = 6  # hard cap; every query costs one LinkedIn search per configured region
# Turn budget PER QUERY. Each check_and_record_job / queue_candidate call burns a turn, so a
# budget shared across all queries silently starves the later ones (see run_scraper).
#
# Sized for the post-2026-08-18 access pattern. Harvesting a page is ONE read-only evaluate, but
# every listing still costs a check_and_record_job (and possibly a queue_candidate) turn, and each
# search now sets its filters by CLICKING CHIPS like a person — location (open, clear, type, pick
# the suggestion), date posted, experience level — which is another ~10-14 turns per search with a
# randomised pause between each. At SCRAPER_MAX_LISTINGS_PER_SEARCH (25) across the usual two
# regions that is ~80 record turns plus ~28 chip turns before overhead. Starvation is silent — the
# model simply stops and the run reports "N listings inspected" with no error — so this is set with
# headroom rather than tuned tight. Turns are cheap here; account safety is not (see the Account
# safety requirement in docs/requirements.md).
SCRAPER_MAX_TURNS_PER_QUERY = 260

# Floor below which a query clearly bailed before finishing one harvest cycle (navigate, snapshot,
# scroll, evaluate, then a record call per listing). Absolute, NOT a fraction of the budget above:
# the budget is sized for the worst case, so a fraction fired on every healthy query once
# harvesting replaced click-to-reveal (a good query legitimately runs ~50 of 180 turns), and a
# warning that fires on success trains the reader to ignore it.
SCRAPER_MIN_TURNS_PER_QUERY = 12

# Human-emulation pacing for Stage 1b.
#
# The scraper drives a REAL logged-in LinkedIn account through the shared Playwright profile.
# Getting that account flagged or banned costs incomparably more than a slow run, so discovery is
# paced rather than run at machine speed.
#
# There is still no per-LISTING delay: the id is read straight off the card's `componentkey`
# attribute, so harvesting is one read-only evaluate with no interaction burst to disguise.
#
# There IS a per-ACTION delay again, because filters are now applied by clicking chips rather than
# by crafting `geoId`/`f_TPR` URL parameters. Hand-assembling filter params is something no human
# ever does and is a cheap fingerprint for anti-automation; clicking the chips is what a person
# does, and a person does not do it instantly or at a metronome-steady rate.
#
# Ranges are (min, max) seconds; a value is drawn uniformly per pause, because a fixed interval is
# itself a robotic signature. Never replace these with constants.
SCRAPER_INTER_ACTION_DELAY_SECONDS = (0.8, 2.6)    # between individual chip/filter interactions
SCRAPER_INTER_SEARCH_DELAY_SECONDS = (2.5, 6.0)    # between searches within one query
SCRAPER_INTER_QUERY_DELAY_SECONDS = (8.0, 18.0)    # between queries (code-enforced, not prompted)

# Listings recorded per search. Now that harvesting is a single read-only call rather than N
# clicks, taking the whole first page costs nothing extra in account exposure — the cap is just a
# sanity bound on how many check_and_record_job turns one search can burn.
SCRAPER_MAX_LISTINGS_PER_SEARCH = 25

# Below this many listings, a query is retried rather than recorded as a real result.
#
# Counted in DISTINCT listings (see run_scraper): call counts double-count a job surfaced by two
# regions, which is what let a collapsed region axis look like healthy coverage.
#
# Sized against what a working query actually yields. One region alone returns ~25 cards, so a
# healthy two-region query lands somewhere between ~17 (heavy legitimate cross-region overlap) and
# 50. The old floor of 5 was calibrated for the pre-chip regime and was far too low: on the first
# live chip run a query that searched only ONE of its two regions returned 6 distinct listings and
# sailed straight through. Must be >= 1 or recovery is disabled.
#
# 12 -> 6 (2026-09-19): at 12, niche queries (e.g. "Agentic AI Engineer") that legitimately return
# 5-11 fell below it and paid for a recovery pass most runs. A half-run is still caught by the
# per-region report_search check below, which does not depend on a count at all — this is a
# backstop, not the primary detector.
SCRAPER_MIN_LISTINGS_PER_QUERY = 6

# Below this many distinct listings AFTER the recovery pass, the query raises a `low_listings`
# ui_alert (Telegram NEEDS ATTENTION). Checked after recovery so the alert means the retry did not
# fix it: checking the first pass fired on 7 of 8 runs, mostly for queries the retry had already
# rescued, and an alert that fires on success trains the reader to ignore it.
SCRAPER_ALERT_MIN_LISTINGS_PER_QUERY = SCRAPER_MIN_LISTINGS_PER_QUERY // 2

# Date-posted filter applied via the "Date posted" chip. LinkedIn expresses the choice as
# f_TPR=r<seconds>; we never navigate to that parameter ourselves (see the chip rationale above),
# but we DO read it back off the URL to confirm the click actually applied. r604800 = past week,
# which suits a daily run and sits well inside the 21-day staleness gate.
SCRAPER_DATE_POSTED_SECONDS = 604800
SCRAPER_DATE_POSTED_LABEL = 'Past week'

# Experience-level chip option. Individual-contributor seniority only: Manager / Director /
# Executive are people-management, which the query generator already excludes by title.
SCRAPER_EXPERIENCE_LABEL = 'Senior'

# --- UI contract -----------------------------------------------------------------------------
#
# LinkedIn restructures this page without notice and keeps adding anti-automation measures. A
# hard-coded reader fails SILENTLY against that: it returns zero listings, which is indistinguish-
# able from "no new jobs today". That is exactly how the 2026-08-11 regression and the dead EU
# region survived for days. So every search asserts the page's structure and says so out loud.
#
# Each entry is a named structural expectation checked against the report returned by
# SCRAPER_UI_CONTRACT_JS. `required` elements failing is a contract VIOLATION (the query stops and
# the user is alerted); non-required ones only feed fingerprint drift.
UI_CONTRACT_ELEMENTS = (
    {'name': 'search_box', 'required': True, 'description': 'the "Describe the job you want" textbox'},
    {'name': 'results_container', 'required': True, 'description': 'div[componentkey="SearchResultsMainContent"]'},
    {'name': 'job_cards', 'required': True, 'description': 'div[componentkey^="job-card-component-ref-"] cards'},
    {'name': 'chip_row', 'required': True, 'description': 'the filter chip row'},
    {'name': 'location_pin', 'required': True, 'description': 'the location chip / pin'},
    {'name': 'pagination', 'required': False, 'description': 'the 1 / 2 / 3 / Next footer'},
)

# Substrings that mean LinkedIn is challenging us rather than that the markup moved. Checked FIRST
# and handled oppositely: a changed selector may be retried, a challenge must NEVER be — pushing
# through a block converts a soft signal into a confirmed evasion pattern.
UI_BLOCK_SIGNATURES = (
    'captcha',
    'unusual activity',
    'verify your identity',
    "let's do a quick security check",
    'security verification',
    'sign in to continue',
    'please sign in',
    'you have been blocked',
    'access to this page has been denied',
)

# Only these chips are STRUCTURAL -- part of the page's shape. The rest of the row is LinkedIn's
# per-query topical suggestions (Gen AI, LLM, AWS, Computer Vision, AI/ML, Analytics...), which
# legitimately change with every query. Fingerprinting the whole row made drift fire on ordinary
# query-to-query variation, and a warning that fires on success trains the reader to ignore it.
#
# A chip is also matched by its CHOSEN value, since choosing one relabels it ("Date posted" ->
# "Past week", "Experience level" -> "Senior").
UI_STRUCTURAL_CHIPS = (
    'Jobs', 'Date posted', 'Experience level', 'Employment type', 'Company',
    'Under 10 applicants', 'In my network', 'Easy Apply',
    'Past month', 'Past week', 'Past 24 hours',
    'Entry-level', 'Senior', 'Manager', 'Director', 'Executive',
)

# Where the last-known-good page shape is remembered, so a change is noticed the run it happens
# rather than months later. Lives in run_dir (gitignored, machine-local run state).
UI_FINGERPRINT_FILENAME = 'ui_fingerprint.yaml'

# --- Yield alerting --------------------------------------------------------------------------
#
# A saturated run and a healthy run looked identical to the user: silence. Aug 15-18 produced 4, 4,
# 2 and 1 new jobs from ~300 listings each with no signal that anything was wrong. These thresholds
# would have fired on all four and stayed quiet on Aug 14 (26 new).
SATURATION_MIN_NEW_RATIO = 0.05   # share of DISTINCT listings never seen before
SATURATION_MIN_NEW_JOBS = 3       # ...or this many new jobs outright, whichever is kinder
# Two regions returning near-identical id sets means the region axis is dead (Aug 18 scored ~1.0).
REGION_OVERLAP_ALERT_THRESHOLD = 0.5


# Playwright MCP tools removed from the scraper's context.
#
# `allowed_tools` does NOT do this: it only auto-grants permission, and under
# permission_mode='bypassPermissions' permission is already granted, so it is inert (measured:
# with allowed_tools set to 4 tools the model still saw all 24). `disallowed_tools` is the only
# option that removes a tool from the model's context.
#
# This list is deliberately conservative. Scraping is NOT a deterministic problem — see the
# "Scrape Job Postings" use case in docs/requirements.md — so anything the model might need to get past a
# changing page, a consent dialog, a login wall, or a bot check stays available. Measured on a
# live LinkedIn search-results page:
#   - browser_evaluate is REQUIRED: scrolling the inner results container lazy-loads more
#     listings (7 -> 10, +43%). Body-level PageDown reveals nothing, so evaluate is the only
#     working scroll. Removing it silently cuts discovery.
#   - browser_click is REQUIRED: the filter chips (location, date posted, experience level) are
#     clicked to apply a search's filters. Nothing in the RESULTS LIST is ever clicked — the only
#     real <button> in a result card is Dismiss (see docs/requirements.md, Account safety).
#   - Kept for obstacle handling even though unused in the happy path: fill_form, type, hover,
#     select_option, handle_dialog, console_messages.
#   - Kept for diagnosing and working around a changed or hostile page: take_screenshot (the
#     only way to SEE a block page, CAPTCHA, or consent overlay that the a11y tree renders
#     uninformatively), find, press_key. These cost turns when the model over-uses them, but
#     the cost of not having them is a silent zero-listing run.
# Removed below: tools with no role in reading a results list, plus run_code_unsafe, which is
# fully covered by browser_evaluate.
SCRAPER_DISALLOWED_BROWSER_TOOLS = [
    'mcp__playwright__browser_file_upload',
    'mcp__playwright__browser_drag',
    'mcp__playwright__browser_drop',
    'mcp__playwright__browser_resize',
    'mcp__playwright__browser_tabs',
    'mcp__playwright__browser_network_request',
    'mcp__playwright__browser_network_requests',
    'mcp__playwright__browser_navigate_back',
    'mcp__playwright__browser_close',
    'mcp__playwright__browser_run_code_unsafe',
]
# Never remove these — measured as load-bearing for discovery (guarded by a unit test).
SCRAPER_REQUIRED_BROWSER_TOOLS = [
    'mcp__playwright__browser_navigate',
    'mcp__playwright__browser_snapshot',
    'mcp__playwright__browser_click',
    'mcp__playwright__browser_wait_for',
    'mcp__playwright__browser_evaluate',
]

# Stage model + route selection.
#
# One const per pipeline stage, following the shared naming pattern
# [<provider_prefix>_]MODEL_NAME_<desc>. The family prefix of the const this points at
# IS the route selector — there is deliberately no separate *_PROVIDER-style string
# knob any more (two encodings of one decision drift apart; the family prefix already
# carries the routing, and models.route_for() dispatches on it):
#   OPENROUTER_*  -> OpenRouter chat API via the MCP server (:8006), then Anthropic fallback
#   ANTHROPIC_*   -> the Anthropic SDK directly
#   OLLAMA_*      -> the local Ollama MCP server (:8002)
# Repoint a const at a different family to change the route; repoint it within a family
# to change only the model.

MODEL_NAME_QUERY = OPENROUTER_MODEL_NAME_DEFAULT_INTELLIGENCE
MODEL_NAME_COMPANY_MATCH = OPENROUTER_MODEL_NAME_DEFAULT_INTELLIGENCE
MODEL_NAME_RATING = OPENROUTER_MODEL_NAME_DEFAULT_INTELLIGENCE
# Reading amounts out of a compensation phrase is a PARSE, not a judgement — the deterministic
# tiers in salary.py answer all but a few strings, and this only sees what they could not read.
# Flash tier for that reason, and cached per string so a phrase is parsed once, ever.
MODEL_NAME_SALARY = OPENROUTER_MODEL_NAME_DEFAULT_AGENTIC
# The extraction loop is a many-iteration tool-calling conversation, so it runs on the
# shared agentic (flash-tier) default rather than the intelligence default — glm-5.2
# measured agentic 45.7 vs 58.2 for glm-5.3-flash, at ~1/19th the per-token price.
MODEL_NAME_EXTRACTOR = OPENROUTER_MODEL_NAME_DEFAULT_AGENTIC
# Judges a LinkedIn job-page section nobody has classified yet: keep or remove. A judgement, so the
# intelligence tier; called only for a section signature never seen before, then cached for good.
MODEL_NAME_PAGE_SECTIONS = OPENROUTER_MODEL_NAME_DEFAULT_INTELLIGENCE
# Stage 1b scraping is the one place DeepSeek wins (implicit caching, no cache-write fee;
# measured 9.3x cheaper than Haiku on one live search — $0.0283 vs $0.4330 — because
# OPENROUTER_MODEL_NAME_SCRAPER caches implicitly at ~88% and cache writes were 42% of
# the Haiku bill). Do NOT repoint it at the intelligence default: glm-5.2 measured
# $0.1932/M cache-read (~2x Haiku's rate, costing MORE than what it replaces), as did
# deepseek-v4-pro and qwen3.8-max. The win is specific to the flash tier. Pointing it at
# an ANTHROPIC_MODEL_NAME_* const instead routes Stage 1b through the original
# ClaudeSDKClient scraper, kept intact as the rollback path.
MODEL_NAME_SCRAPER = OPENROUTER_MODEL_NAME_SCRAPER
# Replaces max_turns for the OpenRouter loop. A healthy search measured 26 iterations; this is sized
# for two regions plus recovery, with headroom, because starvation is silent (see Search coverage).
SCRAPER_OPENROUTER_MAX_ITERATIONS = 90
# One OpenRouter endpoint serves every scraper call of a run (2026-09-24). Within the server's bpw
# pin, OpenRouter otherwise spreads a conversation across ~16 fp8 providers weighted by price, and
# their CACHE-READ prices for this model spanned 0.0017-0.07 $/M: identical runs cost $0.05 or
# $1.59 on the luck of the draw. Endpoints are ranked by the effective price of the scraper's own
# token mix, measured over the runs of 2026-09-11..24: this share of prompt tokens were cache
# reads, and this many completion tokens were generated per prompt token.
SCRAPER_CACHED_PROMPT_SHARE = 0.94
SCRAPER_COMPLETION_TOKENS_PER_PROMPT_TOKEN = 0.002
# Moves to the next-cheapest endpoint after a provider-side failure, per run, before the scraper
# gives up on OpenRouter and falls back to the Anthropic scraper.
SCRAPER_PROVIDER_MAX_SWITCHES = 2
# A query whose effective $/M prompt tokens exceeds the chosen endpoint's estimate by this factor
# raises a run alert: the week this pin fixes cost ~$5-7 extra with nothing flagging it.
SCRAPER_COST_ALERT_FACTOR = 3
# Per tool result. A LinkedIn a11y snapshot is far larger than anything the scraper needs to reason
# about, and uncached every byte is re-billed on every later iteration.
SCRAPER_TOOL_RESULT_MAX_CHARS = 30_000
# browser_find, split out from the cap above (2026-09-24). Unlike a full snapshot it returns only
# the matches the model asked for, so what it cut was signal: it went over the shared cap 13 times
# across all logged runs, by a few hundred chars up to ~26K. Sized above the largest one measured.
SCRAPER_FIND_RESULT_MAX_CHARS = 60_000

# Audit configuration
AUDIT_OPUS_SAMPLE_SIZE = 2  # jobs sampled per un-surfaced pool for --audit-opus

# Stage 2 (evaluation) configuration
EXTRACTOR_OPENROUTER_MAX_ITERATIONS = 10
# Budget for the a11y snapshot on the FALLBACK path only (linkedin_page._read_snapshot, used when
# the DOM capture fails or the description is missing). The normal path sends de-cluttered text,
# measured 3-9K chars, and never needs a cut. One LinkedIn snapshot once lost 34,797 chars from
# its END, where a JD keeps compensation and work-authorization statements (2026-09-22), so the
# overflow is cut from the MIDDLE by truncate_reported_middle rather than from the tail.
EXTRACTOR_SNAPSHOT_MAX_CHARS = 80_000
# Share of a middle-truncated text kept as the head; the rest is the tail. Above half because a
# page's own structure is front-loaded and only the trailing facts need rescuing.
SNAPSHOT_HEAD_SHARE = 0.6
# The `salary` field's contract, defined ONCE and imported by all three extract schemas
# (agent.EXTRACT_OUTPUT_SCHEMA, tools_generic.submit_job_extract, extract_openrouter's tool spec).
# Two of those carried a "must stay in sync" comment and no test; the field itself carried no
# description at all, and a model handed back a rounded, one-ended range that reached the user
# (2026-09-22). Asking for the text verbatim is the courier-argument exception the Anti-fabrication
# NFR allows for submit_job_extract — which is why raw postings are now retained, so it is
# checkable rather than merely requested.
SALARY_FIELD_DESCRIPTION = (
    'Compensation exactly as the posting states it. Copy BOTH ends of a range, the currency and '
    'the period, e.g. "CA$208,580 - CA$273,770 per year" or "€700-€900/day". Never round a figure, '
    'never give only one end of a range the posting states in full, and never convert a currency. '
    'Put bonus/equity wording after the amounts. Leave empty ONLY when the page states no pay at all.'
)
TRIAGE_ENABLED = True
TRIAGE_THRESHOLD = 1  # skip the rating call when local triage scores <= this (clear low fits)
REFERENCE_SUMMARY_MAX_CHARS = 2500
# No output-token cap is set anywhere, deliberately. There used to be LLM_JSON_MAX_TOKENS = 3000,
# which no call site ever chose — all eight inherited it as a default argument — against models
# that allow 131,072 completion tokens (glm-5.3-flash) with no server-side clamp. Every one of the
# 13 truncations across 2026-09-01..03 was `completion_tokens: 3000` exactly and survived a model
# change, and each cost double: the wasted reasoning tokens plus the Anthropic fallback that ran
# after them. If a ceiling is ever genuinely needed, pass `max_tokens` at the call site that needs
# it rather than reinstating a global default nobody reads.

# Browser toolchain (@playwright/mcp) — PINNED, never '@latest'.
#
# Both launch sites used `npx @playwright/mcp@latest`, re-resolved by npm on every run. npx had
# cached `@playwright/mcp: "^0.0.79"` under the key `@playwright/mcp@latest`, and `^0.0.79` on a
# 0.0.x version means EXACTLY 0.0.79 — so when upstream published 0.0.80 the cached tree stopped
# satisfying `latest` and npx halted the run at an interactive `Ok to proceed? (y)`. That is the
# startup path cron uses, where nothing can answer.
#
# The prompt was the symptom; the silent upgrade is the defect. This package is the tool surface
# the Stage 1b scraper drives a REAL logged-in LinkedIn account through, and the --snapshot-mode
# analysis in docs/requirements.md (Cost efficiency) is verified against one specific bundle. `@latest` could invalidate that
# with no commit — the same shape as the `mcp>=1.29` -> 2.0.0 re-resolution under Dependency pinning there.
#
# Upgrading is therefore a reviewable edit to this constant, prompted for by
# check_playwright_mcp_version() rather than taken automatically.
PLAYWRIGHT_MCP_VERSION = '0.0.82'
PLAYWRIGHT_MCP_PACKAGE = f'@playwright/mcp@{PLAYWRIGHT_MCP_VERSION}'
# Queried directly over HTTPS rather than via `npm view`, which shells out through the npm cache
# and can fail for reasons unrelated to the registry (a root-owned cache file, for one).
PLAYWRIGHT_MCP_REGISTRY_URL = 'https://registry.npmjs.org/@playwright/mcp/latest'
# Applied by requests to the connect AND the read separately, so this is a ~6s ceiling, not
# a 3s one; DNS resolution is bounded by neither. The measured cost against a healthy
# registry is ~0.25s. The call fails open, so a slow registry delays a run, never fails it.
PLAYWRIGHT_MCP_VERSION_CHECK_TIMEOUT_SECONDS = 3

# Text budgets. A cap on text is a decision, so each one is named here, and every cut goes through
# text_budget.truncate_reported() so it is logged and marked rather than silent (2026-09-21: the
# PDF categorizer read only the first 3000 raw chars, whitespace included, and said nothing).
# Deterministic checks (regex gates) are never truncated at all: scanning the full text is free.
#
# Model-derived cap for PDF text sent to ANTHROPIC_MODEL_NAME_LOW (claude-haiku-4-5, 200K-token
# context window per https://platform.claude.com/docs/en/about-claude/models/overview). The
# chars-per-token figure is a deliberately low estimate for English prose, so the char cap errs
# toward fitting; the share leaves the rest of the window for the prompt scaffolding and output.
ANTHROPIC_MODEL_LOW_CONTEXT_TOKENS = 200_000
CHARS_PER_TOKEN_ESTIMATE = 3
PDF_PROMPT_CONTEXT_SHARE = 0.5
PDF_PROMPT_MAX_CHARS = int(ANTHROPIC_MODEL_LOW_CONTEXT_TOKENS * CHARS_PER_TOKEN_ESTIMATE * PDF_PROMPT_CONTEXT_SHARE)
# Per reference job, in the full reference block and the summarization prompt. Up to
# MAX_REFERENCE_JOBS of these are concatenated, so this budget is per job, not per prompt.
REFERENCE_JOB_PROMPT_MAX_CHARS = 3000
# The description shown to the blacklist confirmation call alongside company/location/title. Sized
# above the longest description measured across 2,345 saved jobs (2026-09-24), so in practice it is
# whole: the call runs only on an exact name hit, and more of the posting is what tells "Cohere" the
# AI lab from Cohere Health.
BLACKLIST_CONTEXT_DESCRIPTION_MAX_CHARS = 8_000
# OpenAI-compatible function-calling APIs reject a tool description longer than this.
OPENROUTER_TOOL_DESCRIPTION_MAX_CHARS = 1024
# The scraper's report_blocked `what_happened` argument, echoed into logs and the audit.
SCRAPER_WHAT_HAPPENED_MAX_CHARS = 500
# Diagnostic excerpts of a response, error or command in log lines and exception messages.
LOG_SNIPPET_MAX_CHARS = 300
# The salary text shown on the 💰 line of a Telegram job-match message.
NOTIFICATION_SALARY_MAX_CHARS = 200
# Compensation sentences of the description handed to the salary classifier, and ONLY when the
# deterministic tiers found no figure in the salary field itself.
SALARY_CONTEXT_MAX_CHARS = 1500
# A retained raw posting. Generous: the whole point is to be able to check an extracted figure
# against what the page said, and a capture that drops the compensation block answers nothing.
RAW_POSTING_MAX_CHARS = 200_000

# Multipliers for the amount suffixes a posting writes instead of zeros ('60-75K', '€1.2M').
SALARY_THOUSAND_MULTIPLIER = 1_000
SALARY_MILLION_MULTIPLIER = 1_000_000
# A run of exactly this many digits after a '.' or ',' is a thousands separator, not a decimal:
# '208,580' and '392.000' are both whole amounts, '46.50' is not.
THOUSANDS_GROUP_DIGITS = 3

# run_dir/raw_postings/ retention. Long enough to investigate the current run and the one before
# it, short enough that a daily run does not accumulate. Read by nothing in the pipeline — the
# directory exists so an extracted figure can be checked against the page it came from.
RAW_POSTINGS_RETENTION_DAYS = 7

SECONDS_PER_MINUTE = 60
# Waiting for a freshly launched @playwright/mcp to answer: this many polls, this far apart. A
# server that never answers raises rather than being handed back to fail later with no context.
PLAYWRIGHT_MCP_READY_POLL_ATTEMPTS = 30
PLAYWRIGHT_MCP_READY_POLL_INTERVAL_SECONDS = 1
# Stage 2 restarts a dead @playwright/mcp this many times per run, then stops and hands the
# unevaluated jobs back to the next run. On 2026-09-21 its node process died of a V8 heap OOM
# 1h38m in, and the 23 remaining jobs each failed in milliseconds and were lost to dedup.
PLAYWRIGHT_MAX_RESTARTS_PER_RUN = 1

# Run reporting
YIELD_HISTORY_RUNS_SHOWN = 5          # recent runs listed in the yield-history section of alerts
LISTING_ESTIMATE_HISTORY_RUNS = 10    # past runs whose per-query listing counts feed the median estimate
FAILED_QUERIES_NAMED_MAX = 3          # failed queries named in a partial-failure alert before "+N more"
CONSOLE_BANNER_WIDTH = 40
# Two cost figures closer than this are the same number printed twice; only a real gap is shown.
COST_DELTA_DISPLAY_TOLERANCE_USD = 1e-9
# --audit: a gate-killed job the strong rater scores at or above this is reported as a false negative.
AUDIT_FALSE_NEGATIVE_MIN_RATING = 3
# Guess Unbucketed Location: the model's reason, cut at a word boundary for a YAML comment and Telegram.
LOCATION_GUESS_REASON_MAX_CHARS = 150

# Stage 2 extract fallback (Anthropic, deterministic Playwright)
EXTRACT_FALLBACK_MAX_TURNS = 16
EXTRACT_PAGE_RENDER_WAIT_SECONDS = 3  # the job description renders after navigation

# LinkedIn job pages are read by CODE, not by the extractor model (2026-09-24). One read-only
# browser_evaluate serializes the DOM after LinkedIn's own scripts have run; code then removes the
# clutter and hands the model the text. Measured on three pages, the a11y snapshot the model used
# to read was 58K-171K chars, of which the description was ~15%; the rest was nav, upsells, the
# company's social posts and "More jobs" — other companies' listings WITH their salaries.
#
# Read-only: no DOM writes, no .click(), no dispatchEvent. Built without string or regex literals
# that could be escaped in transit (a `'<!'` literal came back as a SyntaxError in the POC).
# `chars` is the JS length, in UTF-16 units, so read_job_page can tell a capture cut in transit.
LINKEDIN_PAGE_CAPTURE_JS = r'''() => {
  const lt = String.fromCharCode(60), nl = String.fromCharCode(10);
  const doctype = document.doctype ? lt + '!DOCTYPE ' + document.doctype.name + '>' + nl : '';
  const html = doctype + document.documentElement.outerHTML;
  return {url: location.href, title: document.title, chars: html.length, html};
}'''
# Sections removed wherever they appear, matched against an element's OWN text from its start.
# Never by class name: LinkedIn's classes are hashed build output and change between deploys.
LINKEDIN_REMOVE_SECTION_PHRASES = (
    'Job search smarter with Premium',
    'Take the next step in your job search',
    'Looking for talent?',
    'People you can reach out to',
    'Set alert for similar jobs',
    'Unlock hiring insights',
    'More jobs',
    'Use AI to assess how you fit',
    'Interested in working with us in the future?',
    'Trending employee content',
)
# Single elements removed by their exact aria-label.
LINKEDIN_REMOVE_ARIA_LABELS = ('Show more about the company',)
# Page chrome removed whole.
LINKEDIN_REMOVE_TAGS = ('header', 'footer', 'nav')
# Retries of the capture when the description has not rendered yet, each after the render wait.
LINKEDIN_PAGE_CAPTURE_RETRIES = 1
# Timeout for fetching the posting company's logo and the page stylesheet. Fetched with plain
# requests — no cookies — from LinkedIn's public CDN, never through the logged-in browser.
JOB_PAGE_ASSET_FETCH_TIMEOUT_SECONDS = 15
# An unknown page section is shown to MODEL_NAME_PAGE_SECTIONS as its text, up to this many chars
# each (a cut is reported), and is cached under a signature of up to this many words of its first line.
PAGE_SECTION_SAMPLE_MAX_CHARS = 1500
PAGE_SECTION_SIGNATURE_MAX_WORDS = 8

# Local LLM sampling: low for the 1-5 triage score and JSON answers, which should be repeatable.
LOCAL_LLM_TEMPERATURE = 0.2
