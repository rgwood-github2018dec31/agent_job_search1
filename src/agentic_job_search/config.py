from utils_tools_n_agents_common.models import (
    OPENROUTER_MODEL_NAME_DEFAULT_AGENTIC,
    OPENROUTER_MODEL_NAME_DEFAULT_INTELLIGENCE,
)

# Job search parameters
JOB_MAX_AGE_DAYS = 21
JOB_STALE_AGE_DAYS = 30  # hard rule: postings older than this are auto-rated 1
MAX_REFERENCE_JOBS = 20
THINKING_MAX_CHARS = 1000
# Applied-job corpus: only records applied to within this window feed query generation,
# the ideal-role profile, and the already-applied company blocklist. Older PDFs are kept
# on disk but never read.
APPLIED_JOBS_HORIZON_DAYS = 90
# Company blacklist: an entry stops rejecting this many days after its `added` date. A standing
# "never show me this company" decision goes stale — six months on, the reason for it may not
# hold any more — so the entry lapses and is reported rather than silently outliving its reason.
COMPANY_BLACKLIST_EXPIRY_DAYS = 180

# NOTE: personal preferences — search regions, acceptable hybrid locations, the hybrid rating cap,
# work-authorization/language/education gates, and target titles — deliberately do NOT live here.
# They are loaded from the gitignored run_dir/preferences.yaml via preferences.py, so this repo
# carries no information about whoever is running it. See preferences.example.yaml.

# Local/remote LLM MCP tool servers (started via their scripts/start-tool-server.sh)
LLM_LOCAL_MCP_URL = 'http://127.0.0.1:8002/mcp'
LLM_OPENROUTER_MCP_URL = 'http://127.0.0.1:8006/mcp'
# Triage is one cheap 1-5 JSON score whose whole point is to be free and fast, so this is sized to
# that rather than to a 20-35B tier: granite4.1:3b measures 65.9 tok/s / 3.9s cold start / 2.1GB on
# this machine and advertises structured JSON output as a first-class capability.
#
# This is a MACHINE-LOCAL Ollama tag and it can vanish without warning: `qwen3.6:latest` sat here
# until a re-pull replaced it with `qwen3.6:27b-mlx`/`35b-mlx`, and Stage 2c triage then failed open
# on every job for ten days while the run log said only "unhandled errors in a TaskGroup". That is
# what preflight_local_model() in triage.py now catches, once per run, naming what IS installed.
LOCAL_MODEL = 'granite4.1:3b'
# Single-call OpenRouter tasks (query generation, company matching, rating): the shared
# intelligence default. Agentic loops (extraction) get their own const below — they want
# the flash tier for cost, not the strongest single-call model.
OPENROUTER_MODEL = OPENROUTER_MODEL_NAME_DEFAULT_INTELLIGENCE

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
# safety requirement in CLAUDE.md).
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
# sailed straight through. 12 is roughly half a single page — comfortably below a legitimately
# overlapping query, comfortably above a half-run. Must be >= 1 or recovery is disabled.
#
# This is a backstop, not the primary detector: a region that never ran is caught precisely by the
# per-region report_search check in run_scraper, which does not depend on a count at all.
SCRAPER_MIN_LISTINGS_PER_QUERY = 12

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
# "Scrape Job Postings" use case in CLAUDE.md — so anything the model might need to get past a
# changing page, a consent dialog, a login wall, or a bot check stays available. Measured on a
# live LinkedIn search-results page:
#   - browser_evaluate is REQUIRED: scrolling the inner results container lazy-loads more
#     listings (7 -> 10, +43%). Body-level PageDown reveals nothing, so evaluate is the only
#     working scroll. Removing it silently cuts discovery.
#   - browser_click is REQUIRED: the filter chips (location, date posted, experience level) are
#     clicked to apply a search's filters. Nothing in the RESULTS LIST is ever clicked — the only
#     real <button> in a result card is Dismiss (see CLAUDE.md, Account safety).
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

# Stage 1b scraper provider.
#
# 'openrouter' drives the browser through a function-calling loop (scrape_openrouter.py) instead of
# the Claude Agent SDK. Measured 2026-08-21 on one live search: $0.0283 vs $0.4330 for identical
# traffic on Haiku (9.3x), because deepseek-v4-flash caches implicitly (~88% hit rate, and NO
# cache-write fee -- cache writes were 42% of the Haiku bill).
#
# Do NOT point this at glm-5.2 measured $0.1932/M cache-read (~2x Haiku's
# rate, costing MORE than what it replaces), as did deepseek-v4-pro and qwen3.8-max.
# The win is specific to the flash tier.
#
# 'anthropic' selects the original ClaudeSDKClient scraper, kept intact as the rollback path.
SCRAPER_PROVIDER = 'openrouter'  # 'openrouter' | 'anthropic'; falls back to Anthropic on failure
SCRAPER_OPENROUTER_MODEL = 'deepseek/deepseek-v4-flash'
# Replaces max_turns for the OpenRouter loop. A healthy search measured 26 iterations; this is sized
# for two regions plus recovery, with headroom, because starvation is silent (see Search coverage).
SCRAPER_OPENROUTER_MAX_ITERATIONS = 90
# Per tool result. A LinkedIn a11y snapshot is far larger than anything the scraper needs to reason
# about, and uncached every byte is re-billed on every later iteration.
SCRAPER_TOOL_RESULT_MAX_CHARS = 30_000

QUERY_PROVIDER = 'openrouter'  # 'openrouter' (glm) | 'anthropic'; falls back to Anthropic on failure
COMPANY_MATCH_PROVIDER = 'openrouter'  # 'openrouter' (glm) | 'anthropic'; falls back to Anthropic on failure

# Audit configuration
AUDIT_OPUS_SAMPLE_SIZE = 2  # jobs sampled per un-surfaced pool for --audit-opus

# Stage 2 (evaluation) configuration
RATING_PROVIDER = 'openrouter'  # 'anthropic' | 'openrouter' | 'ollama'
EXTRACTOR_PROVIDER = 'openrouter'  # 'anthropic' (Haiku agentic session) | 'openrouter' (function-calling loop)
# The extraction loop is a many-iteration tool-calling conversation, so it runs on the
# shared agentic (flash-tier) default rather than OPENROUTER_MODEL — glm-5.2 measured
# agentic 45.7 vs 58.2 for glm-5.3-flash, at ~1/19th the per-token price.
EXTRACTOR_OPENROUTER_MODEL = OPENROUTER_MODEL_NAME_DEFAULT_AGENTIC
EXTRACTOR_OPENROUTER_MAX_ITERATIONS = 10
EXTRACTOR_TOOL_RESULT_MAX_CHARS = 40_000
TRIAGE_ENABLED = True
TRIAGE_THRESHOLD = 1  # skip the rating call when local triage scores <= this (clear low fits)
REFERENCE_SUMMARY_MAX_CHARS = 2500
LLM_JSON_MAX_TOKENS = 3000  # both default models spend tokens on reasoning before the JSON answer

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
# analysis in CLAUDE.md is verified against one specific bundle. `@latest` could invalidate that
# with no commit — the same shape as the `mcp>=1.29` -> 2.0.0 re-resolution in Known diagnoses.
#
# Upgrading is therefore a reviewable edit to this constant, prompted for by
# check_playwright_mcp_version() rather than taken automatically.
PLAYWRIGHT_MCP_VERSION = '0.0.79'
PLAYWRIGHT_MCP_PACKAGE = f'@playwright/mcp@{PLAYWRIGHT_MCP_VERSION}'
# Queried directly over HTTPS rather than via `npm view`, which shells out through the npm cache
# and can fail for reasons unrelated to the registry (a root-owned cache file, for one).
PLAYWRIGHT_MCP_REGISTRY_URL = 'https://registry.npmjs.org/@playwright/mcp/latest'
# Applied by requests to the connect AND the read separately, so this is a ~6s ceiling, not
# a 3s one; DNS resolution is bounded by neither. The measured cost against a healthy
# registry is ~0.25s. The call fails open, so a slow registry delays a run, never fails it.
PLAYWRIGHT_MCP_VERSION_CHECK_TIMEOUT_SECONDS = 3
