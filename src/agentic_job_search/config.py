from datetime import datetime

# Model tier constants
MODEL_NAME_HIGH = 'claude-opus-5'  # audit reference standard only (--audit-opus)
MODEL_NAME_MEDIUM = 'claude-sonnet-5'
MODEL_NAME_LOW = 'claude-haiku-4-5'

# Job search parameters
JOB_SEARCH_START_DATE = datetime(2026, 4, 1)
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
LOCAL_MODEL = 'qwen3.6:latest'
OPENROUTER_MODEL = 'z-ai/glm-5.2'

# Stage 1 (discovery) configuration
MAX_SEARCH_QUERIES = 6  # hard cap; every query costs one LinkedIn search per configured region
# Turn budget PER QUERY. Each check_and_record_job / queue_candidate call burns a turn, so a
# budget shared across all queries silently starves the later ones (see run_scraper).
#
# Sized for the post-2026-08-11 access pattern: every listing must be SELECTED to reveal its job
# id, so one job costs roughly a click + a URL read + check_and_record_job + queue_candidate + a
# pause. At SCRAPER_MAX_LISTINGS_PER_SEARCH (12) across the usual two regions that is ~120 turns
# before overhead. Starvation is silent — the model simply stops and the run reports "N listings
# inspected" with no error — so this is set with headroom rather than tuned tight. Turns are cheap
# here; account safety is not (see the Account safety requirement in CLAUDE.md).
SCRAPER_MAX_TURNS_PER_QUERY = 180

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
# Only page-level pacing remains. There was a per-selection delay when the scraper had to click
# each card to reveal its job id; that design is gone — the id is read straight off the card's
# `componentkey` attribute (see the 2026-08-11 entry in CLAUDE.md), so a search is now one page
# load plus one read-only evaluate, with no interaction burst to disguise.
#
# Ranges are (min, max) seconds; a value is drawn uniformly per pause, because a fixed interval is
# itself a robotic signature. Never replace these with constants.
SCRAPER_INTER_SEARCH_DELAY_SECONDS = (2.5, 6.0)    # between searches within one query
SCRAPER_INTER_QUERY_DELAY_SECONDS = (8.0, 18.0)    # between queries (code-enforced, not prompted)

# Listings recorded per search. Now that harvesting is a single read-only call rather than N
# clicks, taking the whole first page costs nothing extra in account exposure — the cap is just a
# sanity bound on how many check_and_record_job turns one search can burn.
SCRAPER_MAX_LISTINGS_PER_SEARCH = 25

# Below this many listings, a query is retried rather than recorded as a real result.
#
# Healthy runs inspect ~27 listings per query. A single-digit count means the searches did not
# actually run. The trigger used to be `seen == 0`, which missed the 2026-08-11 collapse: every
# query returned exactly ONE listing — the only one the AI-powered UI exposes an id for without
# selecting it — and the run reported itself a success. Must be >= 1 or recovery is disabled.
SCRAPER_MIN_LISTINGS_PER_QUERY = 5


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
#   - browser_click is REQUIRED: the Next button yields a fully fresh page of listings.
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

QUERY_PROVIDER = 'openrouter'  # 'openrouter' (glm) | 'anthropic'; falls back to Anthropic on failure
COMPANY_MATCH_PROVIDER = 'openrouter'  # 'openrouter' (glm) | 'anthropic'; falls back to Anthropic on failure

# Audit configuration
AUDIT_OPUS_SAMPLE_SIZE = 2  # jobs sampled per un-surfaced pool for --audit-opus

# Stage 2 (evaluation) configuration
RATING_PROVIDER = 'openrouter'  # 'anthropic' | 'openrouter' | 'ollama'
EXTRACTOR_PROVIDER = 'openrouter'  # 'anthropic' (Haiku agentic session) | 'openrouter' (function-calling loop)
EXTRACTOR_OPENROUTER_MAX_ITERATIONS = 10
EXTRACTOR_TOOL_RESULT_MAX_CHARS = 40_000
TRIAGE_ENABLED = True
TRIAGE_THRESHOLD = 1  # skip the rating call when local triage scores <= this (clear low fits)
REFERENCE_SUMMARY_MAX_CHARS = 2500
LLM_JSON_MAX_TOKENS = 3000  # both default models spend tokens on reasoning before the JSON answer
