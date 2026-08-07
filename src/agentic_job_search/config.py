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

# Local/remote LLM MCP tool servers (started via their scripts/start-tool-server.sh)
LLM_LOCAL_MCP_URL = 'http://127.0.0.1:8002/mcp'
LLM_OPENROUTER_MCP_URL = 'http://127.0.0.1:8006/mcp'
LOCAL_MODEL = 'qwen3.6:latest'
OPENROUTER_MODEL = 'z-ai/glm-5.2'

# Stage 1 (discovery) configuration
MAX_SEARCH_QUERIES = 6  # hard cap; every query costs two LinkedIn searches (Canada + EU)
# Turn budget PER QUERY. Each check_and_record_job / queue_candidate call burns a turn, so a
# budget shared across all queries silently starves the later ones (see run_scraper).
#
# Every query now runs FOUR searches (Canada + EU, each date-sorted and relevance-sorted), so
# this has to cover roughly double what it did at two searches. Measured at two searches: 42
# turns for 14 listings. Starvation is silent — the model simply stops and the run reports
# "N listings inspected" with no error — so this is set with headroom rather than tuned tight.
SCRAPER_MAX_TURNS_PER_QUERY = 90

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
